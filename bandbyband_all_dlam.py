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
from toolkit import (get_slices_dlambdas, bands_TiO, bands_off_TiO
                     SimpleSpectrum, model_known_lambda,
                     plot_posterior_samples_for_paper)

archive = h5py.File('/Users/bmmorris/git/aesop/notebooks/spectra.hdf5', 'r+')


from json import load, dump

star_temps = load(open('star_temps.json', 'r'))
colors = load(open('colors.json', 'r'))
stars = load(open('stars.json', 'r'))

# Set additional width in angstroms centered on band core,
# used for wavelength calibration
roll_width = 15
bands = bands_TiO#[:-1]
yerr = 0.001
color_error = 2 * 0.003 #0.003
force_refit = False #True

# Set width where fitting will occur
fit_width = 0*u.Angstrom


path = 'bandbyband_dlam_results.json'
if os.path.exists(path) and not force_refit:
    results = load(open(path, 'r'))
else:
    results = dict()

for star in sorted(stars.keys()):
    if star not in results or force_refit:
        star_results = dict()
        phot_temp = star_temps[star]

        comparison_temp_high = star_temps[stars[star][0][0]]
        comparison_temp_low = star_temps[stars[star][1][0]]

        mixture_coefficient = 1 - ((phot_temp - comparison_temp_low) /
                                   (comparison_temp_high - comparison_temp_low))

        mixture_coefficient = np.max([0, mixture_coefficient])

        print(star, phot_temp, comparison_temp_high, comparison_temp_low, mixture_coefficient)

        # Book keeping:
        target_name = star
        comp1_name = stars[star][0][0]
        comp1_time = stars[star][0][1]
        comp2_name = stars[star][1][0]
        comp2_time = stars[star][1][1]

        target_temp = star_temps[target_name]
        comp1_temp = star_temps[comp1_name]
        comp2_temp = star_temps[comp2_name]

        target_color = colors[target_name]
        comp1_color = colors[comp1_name]
        comp2_color = colors[comp2_name]

        # Load spectra from database
        times = list(archive[target_name])

        for time in times[:1]:
            if time not in star_results or force_refit:
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

                source1_dlambdas = [0]*len(bands)
                source2_dlambdas = [0]*len(bands)

                time_results = dict()

                for inds, band in zip(target_slices.wavelength_splits, bands):
                    band_results = dict()
                    R_lambda = (blackbody_lambda(band.core, comp2_temp) /
                                blackbody_lambda(band.core, comp1_temp)).value

                    def random_in_range(min, max):
                        return (max-min)*np.random.rand(1)[0] + min

                    def lnprior(theta):
                        area, f = theta
                        # f_S = area * R_lambda

                        #net_color = (1 - f_S) * target_color + f_S * comp2_color
                        # net_color = (1 - area) * comp1_color + area * comp2_color

                        W_Q = (1 - area)/( area*R_lambda + (1 - area) )
                        W_S = (area * R_lambda)/( area*R_lambda + (1 - area) )
                        net_color = 2.5 * np.log10(W_Q * 10**(comp1_color/2.5) + W_S * 10**(comp2_color/2.5))
                        # print(net_color, target_color)
                        if 0 <= area <= 1 and 0 < f: #  and -1 < dlam < 1:
                            return -0.5 * (net_color - target_color)**2/color_error**2
                        return -np.inf

                    def lnlike(theta, target, source1, source2):
                        area, f = theta
                        model, residuals = model_known_lambda(target, source1, source2,
                                                              mixture_coefficient, area,
                                                              # [i - dlam for i in source1_dlambdas],
                                                              # [i - dlam for i in source2_dlambdas],
                                                              source1_dlambdas, source2_dlambdas,
                                                              band, inds, R_lambda, width=fit_width,
                                                              uncertainty=f)
                        return residuals

                    def lnprob(theta, target, source1, source2):
                        lp = lnprior(theta)
                        if not np.isfinite(lp):
                            return -np.inf

                        # print(lp, lnlike(theta, target, source1, source2))
                        return lp + lnlike(theta, target, source1, source2)

                    ndim, nwalkers = 2, 6

                    pos = []

                    counter = -1
                    while len(pos) < nwalkers:
                        realization = [random_in_range(0, 1), random_in_range(0.01, 1)]# ,
                                       #random_in_range(-0.01, 0.01)]
                        if np.isfinite(lnprior(realization)):
                            pos.append(realization)

                    sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=8,
                                              args=(target_slices, source1_slices,
                                                    source2_slices))
                    p0 = sampler.run_mcmc(pos, 1000)[0]
                    sampler.reset()

                    sampler.run_mcmc(p0, 2000)
                    sampler.pool.close()
                    samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))

                    #samples[:, 0] *= R_lambda

                    lower, m, upper = np.percentile(samples[:, 0], [16, 50, 84])
                    band_results['f_S_lower'] = m - lower
                    band_results['f_S'] = m
                    band_results['f_S_upper'] = upper - m
                    band_results['yerr'] = np.median(samples[:, 1])

                    corner(samples, labels=['$f_S$', '$f$'])#, '$\Delta \lambda$'])
                    plt.savefig('plots/{0}_{1}_{2}.pdf'.format(star, int(band.core.value),
                                                                   time.replace(':', '_')),
                                bbox_inches='tight')
                    plt.close()

                    fig, ax = plot_posterior_samples_for_paper(samples, target_slices, source1_slices,
                                                               source2_slices, mixture_coefficient,
                                                               source1_dlambdas, source2_dlambdas,
                                                               band, inds, fit_width, star, R_lambda)

                    # fig, ax = plot_posterior_samples(samples, target_slices, source1_slices,
                    #                                  source2_slices, mixture_coefficient,
                    #                                  source1_dlambdas, source2_dlambdas,
                    #                                  band, inds, fit_width, star)
                    plt.savefig('plots/{0}_{1}_{2}_fit.pdf'.format(star, int(band.core.value),
                                                                   time.replace(':', '_')),
                                bbox_inches='tight')
                    plt.close()

                    time_results[int(band.core.value)] = band_results

                star_results[time] = time_results

                results[star] = star_results

                dump(results, open(path, 'w'), indent=4, sort_keys=True)
# Old sigmaDraconis
#