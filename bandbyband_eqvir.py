import matplotlib
matplotlib.use('Agg')   # Solves tkagg problem
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from emcee import EnsembleSampler
# from emcee.mpi_pool import MPIPool
# import sys
import os
import h5py
from corner import corner
from astropy.modeling.blackbody import blackbody_lambda
from toolkit import (get_slices_dlambdas, bands_TiO,
                     SimpleSpectrum, model_known_lambda,
                     plot_posterior_samples_for_paper)

archive = h5py.File('/Users/bmmorris/git/aesop/notebooks/spectra.hdf5', 'r+')

# eqvir_comparisons = [['GJ705', '2018-06-01T05:31:48.119'],  # Photosphere template
#                      ['HD209290', '2016-09-18T06:05:55.740']]  # Spot template
# eqvir_comparisons = [['GJ705', '2018-06-01T05:31:48.119'],  # Photosphere template
#                      ['GJ4099', '2017-09-05T06:43:40.051']]  # Spot template
# eqvir_comparisons = [['HD6497', "2017-11-06T08:38:56.641"],  # Photosphere template
#                      ['GJ4099', '2017-09-05T06:43:40.051']]  # Spot template
eqvir_comparisons = [['GJ9781A', '2016-09-18T06:31:41.670'],  # Photosphere template
                     ['GJ4099', '2017-09-05T06:43:40.051']]  # Spot template

star = 'EQVir'

stars = {star: eqvir_comparisons}

from json import load, dump

star_temps = load(open('star_temps.json', 'r'))
colors = load(open('colors.json', 'r'))

# Set additional width in angstroms centered on band core,
# used for wavelength calibration
roll_width = 10
bands = bands_TiO#[:-1]
yerr = 0.001
force_refit = True

# Set width where fitting will occur
fit_width = 0*u.Angstrom


path = 'bandbyband_{0}_results.json'.format(star)
if os.path.exists(path) and not force_refit:
    results = load(open(path, 'r'))
else:
    results = dict()

phot_temp = star_temps[star]

comparison_temp_high = star_temps[stars[star][0][0]]
comparison_temp_low = star_temps[stars[star][1][0]]

mixture_coefficient = 1 - ((phot_temp - comparison_temp_low) /
                           (comparison_temp_high - comparison_temp_low))

mixture_coefficient = np.max([0, mixture_coefficient])

print(phot_temp, comparison_temp_high, comparison_temp_low, mixture_coefficient)

# Book keeping:
target_name = star
comp1_name = stars[star][0][0]
comp1_time = stars[star][0][1]
comp2_name = stars[star][1][0]
comp2_time = stars[star][1][1]

target_temp = star_temps[target_name]
comp1_temp = star_temps[comp1_name]
comp2_temp = star_temps[comp2_name]

color_error = 1#0.04
target_color = colors[target_name]
comp1_color = colors[comp1_name]
comp2_color = colors[comp2_name]


# Load spectra from database
times = list(archive[target_name])

for time in times:
    if time not in results or force_refit:
        spectrum1 = archive[target_name][time]

        spectrum2 = archive[comp1_name][comp1_time]
        spectrum3 = archive[comp2_name][comp2_time]

        wavelength1 = spectrum1['wavelength'][:]
        flux1 = spectrum1['flux'][:]

        wavelength2 = spectrum2['wavelength'][:]
        flux2 = spectrum2['flux'][:]

        wavelength3 = spectrum3['wavelength'][:]
        flux3 = spectrum3['flux'][:]

        target = SimpleSpectrum(wavelength1, flux1, dispersion_unit=u.Angstrom)
        source1 = SimpleSpectrum(wavelength2, flux2, dispersion_unit=u.Angstrom)
        source2 = SimpleSpectrum(wavelength3, flux3, dispersion_unit=u.Angstrom)

        # Slice the spectra into chunks centered on each TiO band:
        slicesdlambdas = get_slices_dlambdas(bands, roll_width, target, source1, source2)
        target_slices, source1_slices, source2_slices, source1_dlambdas, source2_dlambdas = slicesdlambdas

        time_results = dict()

        for inds, band in zip(target_slices.wavelength_splits, bands):
            band_results = dict()
            R_lambda = (blackbody_lambda(band.core, comp1_temp) /
                        blackbody_lambda(band.core, comp2_temp)).value

            def random_in_range(min, max):
                return (max-min)*np.random.rand(1)[0] + min

            def lnprior(theta):
                area, f = theta
                f_S = area * R_lambda

                # net_color = (1 - f_S) * comp1_color + f_S * comp2_color
                net_color = (1 - f_S) * target_color + f_S * comp2_color

                if 0 <= f_S <= 1 and 0 < f: #lnf < 0:
                    return -0.5 * (net_color - target_color)**2/color_error**2
                return -np.inf

            def lnlike(theta, target, source1, source2):
                area, f = theta
                model, residuals = model_known_lambda(target, source1, source2,
                                                      mixture_coefficient,
                                                      area, source1_dlambdas,
                                                      source2_dlambdas, band, inds,
                                                      width=fit_width,
                                                      uncertainty=f)
                return residuals

            def lnprob(theta, target, source1, source2):
                lp = lnprior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + lnlike(theta, target, source1, source2)

            ndim, nwalkers = 2, 10

            pos = []

            counter = -1
            while len(pos) < nwalkers:
                realization = [random_in_range(0, 1), random_in_range(0.01, 1)]
                if np.isfinite(lnprior(realization)):
                    pos.append(realization)

            # pool = MPIPool(loadbalance=True)
            # if not pool.is_master():
            #     pool.wait()
            #     sys.exit(0)

            sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=8,
                                      args=(target_slices, source1_slices,
                                            source2_slices))
                                      #pool=pool)

            sampler.run_mcmc(pos, 2000)

            samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))


            samples[:, 0] *= R_lambda

            f_S = samples[:, 0]

            net_color = (1 - f_S) * target_color + f_S * comp2_color

            lower, m, upper = np.percentile(samples[:, 0], [16, 50, 84])
            band_results['f_S_lower'] = m - lower
            band_results['f_S'] = m
            band_results['f_S_upper'] = upper - m
            band_results['yerr'] = np.median(samples[:, 1])

            samples = np.hstack([samples, net_color[:, np.newaxis]])

            fig, ax = plt.subplots(3, 3)
            corner(samples, fig=fig, labels=['$f_S$', '$f$', '$c$'])

            axis = ax[-1, -1]
            print(target_color, comp1_color, comp2_color)
            axis.axvline(target_color, color='k', zorder=100)
            axis.axvline(comp1_color, color='b', zorder=100)
            axis.axvline(comp2_color, color='r', zorder=100)
            axis.set_xlim([comp1_color-0.1, comp2_color+0.1])

            plt.savefig('plots_eqvir/{0}_{1}_{2}.pdf'.format(star, int(band.core.value),
                                                           time.replace(':', '_').replace('.', '_')),
                        bbox_inches='tight')
            plt.close()

            fig, ax = plot_posterior_samples_for_paper(samples, target_slices, source1_slices,
                                                       source2_slices, mixture_coefficient,
                                                       source1_dlambdas, source2_dlambdas,
                                                       band, inds, fit_width, star)
            plt.savefig('plots_eqvir/{0}_{1}_{2}_fit.pdf'.format(star, int(band.core.value),
                                                           time.replace(':', '_').replace('.', '_')),
                        bbox_inches='tight')
            plt.close()
            # fig.close()

            time_results[int(band.core.value)] = band_results

        results[time] = time_results

        dump(results, open(path, 'w'), indent=4, sort_keys=True)

