import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from emcee import EnsembleSampler

import h5py
from corner import corner
from astropy.modeling.blackbody import blackbody_lambda
from toolkit import (slice_spectrum, concatenate_spectra, bands_TiO,
                     instr_model, SimpleSpectrum, model_known_lambda)

archive = h5py.File('/Users/bmmorris/git/aesop/notebooks/spectra.hdf5', 'r+')

stars = {'HD122120': [['HD127506', '2017-06-15T04:44:33.939'],
                      ['HD148467', '2017-06-20T06:08:51.240']]}

from json import load

star_temps = load(open('star_temps.json', 'r'))

# Set additional width in angstroms centered on band core,
# used for wavelength calibration
width = 10
bands = bands_TiO
yerr = 0.001

# Set width where fitting will occur
fit_width = 3*u.Angstrom

results = dict()




for star in stars:
    phot_temp = star_temps[star]

    comparison_temp_high = star_temps[stars[star][0][0]]
    comparison_temp_low = star_temps[stars[star][1][0]]

    mixture_coefficient = ((phot_temp - comparison_temp_low) /
                           (comparison_temp_high - comparison_temp_low))

    # Book keeping:
    target_name = star
    comp1_name = stars[star][0][0]
    comp1_time = stars[star][0][1]
    comp2_name = stars[star][1][0]
    comp2_time = stars[star][1][1]

    target_temp = star_temps[target_name]
    comp1_temp = star_temps[comp1_name]
    comp2_temp = star_temps[comp2_name]

    # Load spectra from database
    times = list(archive[target_name])
    time = times[0]
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
    spec_band = []
    for band in bands:
        target_slice = slice_spectrum(target, band.min-width*u.Angstrom,
                                      band.max+width*u.Angstrom)
        target_slice.flux /= target_slice.flux.max()
        spec_band.append(target_slice)

    target_slices = concatenate_spectra(spec_band)

    spec_band = []
    for band, inds in zip(bands, target_slices.wavelength_splits):
        target_slice = slice_spectrum(source1, band.min-width*u.Angstrom,
                                      band.max+width*u.Angstrom,
                                      force_length=abs(np.diff(inds))[0])
        target_slice.flux /= np.percentile(target_slice.flux, 98)
        spec_band.append(target_slice)

    source1_slices = concatenate_spectra(spec_band)

    spec_band = []
    for band, inds in zip(bands, target_slices.wavelength_splits):
        target_slice = slice_spectrum(source2, band.min-width*u.Angstrom,
                                      band.max+width*u.Angstrom,
                                      force_length=abs(np.diff(inds))[0])
        target_slice.flux /= np.percentile(target_slice.flux, 98)
        spec_band.append(target_slice)

    source2_slices = concatenate_spectra(spec_band)

    # Update the wavelength solution for each TiO band slice
    n_bands = len(bands)

    init_model = instr_model(target_slices, source1_slices, source2_slices, 0,
                             *[0]*n_bands)[0]

    source1_dlambdas = []
    for i, inds in enumerate(target_slices.wavelength_splits):
        min_ind, max_ind = inds

        chi2s = []
        for j in range(len(init_model[min_ind:max_ind])):
            chi2s.append(np.sum((target_slices.flux[min_ind:max_ind] -
                                 np.roll(init_model[min_ind:max_ind], j))**2))

        dlambda = np.median(np.diff(target_slices.wavelength[min_ind:max_ind])) * np.argmin(chi2s)
        source1_dlambdas.append(-dlambda.value)

    init_model = instr_model(target_slices, source1_slices, source2_slices, 1,
                             *[0]*n_bands)[0]
    source2_dlambdas = []
    for i, inds in enumerate(target_slices.wavelength_splits):
        min_ind, max_ind = inds

        chi2s = []
        for j in range(len(init_model[min_ind:max_ind])):
            chi2s.append(np.sum((target_slices.flux[min_ind:max_ind] -
                                 np.roll(init_model[min_ind:max_ind], j))**2))

        dlambda = np.median(np.diff(target_slices.wavelength[min_ind:max_ind])) * np.argmin(chi2s)
        source2_dlambdas.append(-dlambda.value)

    for inds, band in zip(target_slices.wavelength_splits, bands):
        R_lambda = (blackbody_lambda(band.core, comp1_temp) /
                    blackbody_lambda(band.core, comp2_temp)).value

        def random_in_range(min, max):
            return (max-min)*np.random.rand(1)[0] + min

        def lnprior(theta):
            area, f = theta
            if 0 < area < 1 and 0 < f: #lnf < 0:
                return 0.0
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

        sampler = EnsembleSampler(nwalkers, ndim, lnprob, threads=8,
                                  args=(target_slices, source1_slices,
                                        source2_slices))

        sampler.run_mcmc(pos, 1000)

        samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

        samples[:, 0] *= R_lambda

        corner(samples, labels=['$f_S$', '$f$'])
        plt.savefig('plots/{0}_{1}.pdf'.format(star, int(band.core.value)),
                    bbox_inches='tight')
        plt.close()