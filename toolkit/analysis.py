from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from scipy.signal import gaussian
from copy import deepcopy
import matplotlib.pyplot as plt

from .spectra import SimpleSpectrum, slice_spectrum, concatenate_spectra

__all__ = ["instr_model", "combine_spectra", "match_spectra", "get_slices_dlambdas",
           'model_known_lambda', 'plot_posterior_samples', "plot_posterior_samples_for_paper"]

err_bar = 0.025


def gaussian_kernel(M, std):
    g = gaussian(M, std)

    if np.all(g == 0):
        if len(g) % 2 == 0:
           g[len(g)//2-1] = 1
           g[len(g)//2] = 1
        else:
           g[len(g)//2] = 1

    return g


def instr_model(observed_spectrum, model_phot, model_spot, spotted_area,
                *lam_offsets):#,
                #res, *lam_offsets, err_bar=err_bar):
                # lam_offset, res, err_bar=err_bar):

    # Kernel for instrumental/rotational broadening profile:
    #kernel_0 = gaussian_kernel(50, res)
    # observed_spectrum = deepcopy(observed_spectrum)
    # observed_spectrum.convolve(kernel=kernel_0)

    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area, 1)
    # combined_spectrum.convolve(kernel=kernel_0)

    # Apply wavelength correction just to red wavelengths:
    corrected_wavelengths = observed_spectrum.wavelength.copy()

    if len(lam_offsets) > 0:
        for i, inds in enumerate(observed_spectrum.wavelength_splits):
            min_ind, max_ind = inds
            corrected_wavelengths[min_ind:max_ind] -= lam_offsets[i]*u.Angstrom

    #corrected_wavelengths = u.Quantity(corrected_wavelengths, u.Angstrom)

    combined_interp = combined_spectrum.interpolate(corrected_wavelengths)
    # combined_interp = combined_spectrum.interpolate(observed_spectrum.wavelength -
    #                                                 lam_offset*u.Angstrom)

    combined_scaled = combined_interp.copy()
    residuals = 0
    for i_min, i_max in observed_spectrum.wavelength_splits:
        # c, residuals_i = np.linalg.lstsq(combined_interp[i_min:i_max, np.newaxis],
        #                                  observed_spectrum.flux[i_min:i_max, np.newaxis])[0:2]
        # residuals += residuals_i**2

        #a = combined_interp[i_min:i_max, np.newaxis]
        mean_wavelength = np.mean(corrected_wavelengths[i_min:i_max]).value
        lam = corrected_wavelengths[i_min:i_max].value - mean_wavelength
        a = np.vstack([combined_interp[i_min:i_max], lam]).T

        b = observed_spectrum.flux[i_min:i_max]
        c = np.linalg.inv(a.T @ a) @ a.T @ b  # Ordinary least squares

        residuals += np.sum((a @ c - b)**2)

        #combined_scaled[i_min:i_max] = combined_interp[i_min:i_max] * c#[0]
        combined_scaled[i_min:i_max] = a @ c

   # residuals /= np.exp(lnf)**2 #err_bar**2

    return combined_scaled, residuals


def model_known_lambda(observed_spectrum, model_phot, model_spot, mixture_coeff,
                       spotted_area, lam_offsets_0, lam_offsets_1, band,
                       input_inds, R, width=1*u.Angstrom, uncertainty=None,
                       diagnostic=False):

    model_phot = deepcopy(model_phot)
    model_spot = deepcopy(model_spot)

    # Apply wavelength correction just to red wavelengths:

    for spectrum, lam_offsets in zip([model_phot, model_spot],
                                     [lam_offsets_0, lam_offsets_1]):
        corrected_wavelengths = spectrum.wavelength.copy()

        # for i, inds in enumerate(spectrum.wavelength_splits):
        #     min_ind, max_ind = inds
        #     corrected_wavelengths[min_ind:max_ind] -= lam_offsets[i]*u.Angstrom

        for i, inds in enumerate(spectrum.wavelength_splits):
            min_ind, max_ind = inds
            # if i == 0:
            #     offset = 0
            # else:
            #     offset = 0.5
            corrected_wavelengths[min_ind:max_ind] -= lam_offsets[i]*u.Angstrom

        # for i, inds in enumerate(spectrum.wavelength_splits):
        #     spectrum.flux = np.roll(spectrum.flux, -lam_offsets[i])
        #
        spectrum.wavelength = corrected_wavelengths.copy()

    # Create a mixture of the hotter and cooler atmospheres with the correct
    # effective temperature for the photosphere of the star
    phot_mixture = combine_spectra(model_phot, model_spot, mixture_coeff, R)

    # Then combine the composite template photosphere model with some more
    # of the cooler star component to represent starspot coverage
    combined_spectrum = combine_spectra(phot_mixture, model_spot, spotted_area, R)

    combined_interp = combined_spectrum.interpolate(observed_spectrum.wavelength)# -

    combined_scaled = combined_interp.copy()

    i_min, i_max = input_inds

    in_range = ((observed_spectrum.wavelength[i_min:i_max] < band.max + width/2) &
                (observed_spectrum.wavelength[i_min:i_max] > band.min - width/2))

    # mean_wavelength = np.mean(observed_spectrum.wavelength[i_min:i_max][in_range]).value
    lam = observed_spectrum.wavelength[i_min:i_max].value - band.core.value#mean_wavelength
    a = np.vstack([combined_interp[i_min:i_max][in_range]]).T
    a_all = np.vstack([combined_interp[i_min:i_max]]).T

    # a = np.vstack([combined_interp[i_min:i_max][in_range], lam[in_range]]).T
    # a_all = np.vstack([combined_interp[i_min:i_max], lam]).T #

    b = observed_spectrum.flux[i_min:i_max][in_range]

    Omega = np.diag(uncertainty**2 * np.ones_like(lam[in_range]))
    inv_Omega = np.linalg.inv(Omega)
    if diagnostic:
        print(a, inv_Omega, b)
    c = np.linalg.inv(a.T @ inv_Omega @ a) @ a.T @ inv_Omega @ b  # Ordinary least squares

    residuals = -0.5*np.sum((a @ c - b)**2/uncertainty**2 + np.log(uncertainty**2))# + np.log(2*np.pi))

    combined_scaled[i_min:i_max] = a_all @ c

    return combined_scaled, residuals


def match_spectra(observed_spectrum, comparison_spectrum,
                  res, *lam_offsets, err_bar=err_bar):
                # lam_offset, res, err_bar=err_bar):

    # Kernel for instrumental/rotational broadening profile:
    kernel_0 = gaussian_kernel(50, res)

    comparison_spectrum = deepcopy(comparison_spectrum)
    comparison_spectrum.convolve(kernel=kernel_0)

    # Apply wavelength correction just to red wavelengths:
    corrected_wavelengths = observed_spectrum.wavelength.copy()

    for i, inds in enumerate(observed_spectrum.wavelength_splits):
        min_ind, max_ind = inds
        corrected_wavelengths[min_ind:max_ind] -= lam_offsets[i]*u.Angstrom

    comparison_interp = comparison_spectrum.interpolate(corrected_wavelengths)

    comparison_scaled = comparison_interp.copy()
    residuals = 0
    for i_min, i_max in observed_spectrum.wavelength_splits:
        a = comparison_interp[i_min:i_max, np.newaxis]
        b = observed_spectrum.flux[i_min:i_max]
        c = np.linalg.inv(a.T @ a) @ a.T @ b  # Ordinary least squares

        residuals += np.sum((a @ c - b)**2)

        comparison_scaled[i_min:i_max] = comparison_interp[i_min:i_max] * c#[0]

    return comparison_scaled, residuals

def combine_spectra(spectrum_phot, spectrum_spot, spotted_area, R):
    spectrum_spot_interp = np.interp(spectrum_phot.wavelength.value,
                                     spectrum_spot.wavelength.value,
                                     spectrum_spot.flux)
    # Eqn 1 of ONeal 2004
    combined_flux = ((spectrum_phot.flux * (1 - spotted_area) +
                     spectrum_spot_interp * spotted_area * R) /
                     (spotted_area*R + (1-spotted_area)))
    return SimpleSpectrum(spectrum_phot.wavelength, combined_flux,
                          dispersion_unit=spectrum_phot.dispersion_unit)


def plot_posterior_samples(samples, target_slices, source1_slices, source2_slices, mixture_coefficient,
                           source1_dlambdas, source2_dlambdas, band, inds, fit_width, star):
    yerr_eff = np.median(samples[:, 1])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    nospots, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, 0,
                                             source1_dlambdas, source2_dlambdas, band, inds,
                                             width=fit_width, uncertainty=yerr_eff)
    allspots, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, 1,
                                              source1_dlambdas, source2_dlambdas, band, inds,
                                              width=fit_width, uncertainty=yerr_eff)
    min_ind, max_ind = inds

    for j in range(2):
        ax[j].errorbar(target_slices.wavelength[min_ind:max_ind].value,
                          target_slices.flux[min_ind:max_ind],
                          yerr_eff*np.ones_like(target_slices.flux[min_ind:max_ind]),
                          fmt='o', color='k', zorder=-100, alpha=1 if j else 0.1)
                       #0.025*np.ones(max_ind-min_ind), fmt='.')
    #     ax[i].plot(target_slices.wavelength[min_ind:max_ind],
    #                best_model[min_ind:max_ind], color='r')

        ax[j].plot(target_slices.wavelength[min_ind:max_ind],# + dlam1*u.Angstrom,
                   nospots[min_ind:max_ind], color='C1', lw=2, ls='--', zorder=10)#, color='r')
        ax[j].plot(target_slices.wavelength[min_ind:max_ind],# + dlam2*u.Angstrom,
                   allspots[min_ind:max_ind],  color='C2', lw=2, zorder=10)#, color='r')

    ax[0].axvspan((band.min-fit_width/2).value, (band.max+fit_width/2).value, alpha=0.05, color='k')
    ax[0].set_xlim([(band.min.value-10), (band.max.value+10)])

    # ax[0].set_xlim([target_slices.wavelength[min_ind].value,
    #                target_slices.wavelength[max_ind-1].value])

    ax[1].set_xlim([(band.min-fit_width/2).value, (band.max+fit_width/2).value])

#     ax[i, 1].set_ylim([0.99*rand_model[min_ind:max_ind].min(), 1.01*rand_model[min_ind:max_ind].max()])

    in_range = ((band.min-fit_width/2 < target_slices.wavelength[min_ind:max_ind]) &
               (band.max+fit_width/2 > target_slices.wavelength[min_ind:max_ind]))

    ax[0].set_ylim([0.5*target_slices.flux[min_ind:max_ind][in_range].min(),
                       1.2*target_slices.flux[min_ind:max_ind][in_range].max()])

    ax[1].set_ylim([0.5*target_slices.flux[min_ind:max_ind][in_range].min(),
                       1.2*target_slices.flux[min_ind:max_ind][in_range].max()])

    plt.setp(ax[0].get_xticklabels(), rotation=20, ha='right')
    plt.setp(ax[1].get_xticklabels(), rotation=20, ha='right')

    n_random_draws = 100
    # draw models from posteriors
    for j in range(n_random_draws):
        step = np.random.randint(0, samples.shape[0])
        random_step = samples[step, 0]
        try:

            rand_model, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, random_step,#np.exp(random_step),
                                                        source1_dlambdas, source2_dlambdas, band, inds,
                                                        width=fit_width, uncertainty=yerr_eff)
        except np.linalg.linalg.LinAlgError:
            pass
        for i, inds in enumerate(target_slices.wavelength_splits):
            min_ind, max_ind = inds
            for j in range(2):
                ax[1].plot(target_slices.wavelength[min_ind:max_ind],
                              rand_model[min_ind:max_ind], color='#389df7', alpha=0.1)#, zorder=10)
    fig.subplots_adjust(hspace=0.5)
    return fig, ax

def plot_posterior_samples_for_paper(samples, target_slices, source1_slices, source2_slices, mixture_coefficient,
                                     source1_dlambdas, source2_dlambdas, band, inds, fit_width, star, R):
    yerr_eff = np.median(samples[:, 1])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))# , sharey=True)

    # slope_fit_width
    slope_fit_width = 10 * u.Angstrom
    nospots, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, 0,
                                             source1_dlambdas, source2_dlambdas, band, inds, R,
                                             width=slope_fit_width, uncertainty=yerr_eff)
    allspots, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, 1,
                                              source1_dlambdas, source2_dlambdas, band, inds, R,
                                              width=slope_fit_width, uncertainty=yerr_eff)
    min_ind, max_ind = inds

    for j in range(2):
        ax[j].errorbar(target_slices.wavelength[min_ind:max_ind].value,
                          target_slices.flux[min_ind:max_ind],
                          yerr_eff*np.ones_like(target_slices.flux[min_ind:max_ind]),
                          fmt='o', color='k', zorder=-100, alpha=1, ecolor='silver')# if j else 0.1)
                       #0.025*np.ones(max_ind-min_ind), fmt='.')
    #     ax[i].plot(target_slices.wavelength[min_ind:max_ind],
    #                best_model[min_ind:max_ind], color='r')

        if j == 0:
            ax[j].plot(target_slices.wavelength[min_ind:max_ind],# + dlam1*u.Angstrom,
                       nospots[min_ind:max_ind], #/np.percentile(nospots[min_ind:max_ind], 98),
                       color='DodgerBlue', lw=1, zorder=10, label='HD 6497')#, color='r')
            ax[j].plot(target_slices.wavelength[min_ind:max_ind],# + dlam2*u.Angstrom,
                       allspots[min_ind:max_ind], #/np.percentile(allspots[min_ind:max_ind], 98),
                       color='r', lw=1, zorder=10, label='GJ 4099')#, color='r')
    ax[0].legend(loc='lower left')
    ax[0].axvspan((band.min-fit_width/2).value, (band.max+fit_width/2).value, alpha=0.05, color='k')
    ax[0].set_xlim([(band.min.value-5), (band.max.value+5)])
    ax[0].set(xlabel="Wavelength [$\AA$]", ylabel='Flux')
    # ax[0].set_xlim([target_slices.wavelength[min_ind].value,
    #                target_slices.wavelength[max_ind-1].value])

    ax[1].set_xlim([(band.min-fit_width/2).value, (band.max+fit_width/2).value])

#     ax[i, 1].set_ylim([0.99*rand_model[min_ind:max_ind].min(), 1.01*rand_model[min_ind:max_ind].max()])

    in_range = ((band.min-fit_width/2 < target_slices.wavelength[min_ind:max_ind]) &
               (band.max+fit_width/2 > target_slices.wavelength[min_ind:max_ind]))

    ax[0].set_ylim([(allspots[min_ind:max_ind]/np.percentile(allspots[min_ind:max_ind], 98)).min(),
                    1.01])

    # ax[1].set_ylim([0.5*target_slices.flux[min_ind:max_ind][in_range].min(),
    #                    1.2*target_slices.flux[min_ind:max_ind][in_range].max()])

    ax[1].set(xlabel='Wavelength [$\AA$]')

    plt.setp(ax[0].get_xticklabels(), rotation=20, ha='right')
    plt.setp(ax[1].get_xticklabels(), rotation=20, ha='right')

    n_random_draws = 100
    # draw models from posteriors
    for k in range(n_random_draws):
        step = np.random.randint(0, samples.shape[0])
        random_step = samples[step, 0]
        rand_model, residuals = model_known_lambda(target_slices, source1_slices, source2_slices, mixture_coefficient, random_step,
                                                    source1_dlambdas,  source2_dlambdas, band, inds,
                                                    R, width=fit_width, uncertainty=yerr_eff)

        ax[1].plot(target_slices.wavelength[min_ind:max_ind],
                      rand_model[min_ind:max_ind], color='gray', alpha=0.05)#,
    ax[1].set_ylim([0.7, 1.1])
    for axis in fig.axes:
        for s in ['right', 'top']:
            axis.spines[s].set_visible(False)
        axis.grid(ls=':')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    return fig, ax


def get_slices_dlambdas(bands, width, target, source1, source2):
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

    init_model_0 = instr_model(target_slices, source1_slices, source2_slices, 0,
                             *[0]*n_bands)[0]

    source1_dlambdas = []
    for i, inds in enumerate(target_slices.wavelength_splits):
        min_ind, max_ind = inds

        corr = np.correlate(init_model_0[min_ind:max_ind] - init_model_0[min_ind:max_ind].mean(),
                            target_slices.flux[min_ind:max_ind] - target_slices.flux[min_ind:max_ind].mean(), mode='full')

        argmax = len(init_model_0[min_ind:max_ind]) - np.argmax(corr)

        dlam = target_slices.wavelength[min_ind:max_ind][1] - target_slices.wavelength[min_ind:max_ind][0]
        dlambda = dlam * argmax
        source1_dlambdas.append(-dlambda.value)

    init_model_1 = instr_model(target_slices, source1_slices, source2_slices, 1,
                               *[0]*n_bands)[0]
    source2_dlambdas = []
    for i, inds in enumerate(target_slices.wavelength_splits):
        min_ind, max_ind = inds

        corr = np.correlate(init_model_1[min_ind:max_ind] - init_model_1[min_ind:max_ind].mean(),
                            target_slices.flux[min_ind:max_ind] - target_slices.flux[min_ind:max_ind].mean(), mode='full')

        argmax = len(init_model_1[min_ind:max_ind]) - np.argmax(corr)

        dlam = target_slices.wavelength[min_ind:max_ind][1] - target_slices.wavelength[min_ind:max_ind][0]
        dlambda = dlam * argmax
        source2_dlambdas.append(-dlambda.value)

    return target_slices, source1_slices, source2_slices, source1_dlambdas, source2_dlambdas
