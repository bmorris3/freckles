from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from scipy.signal import gaussian
from copy import deepcopy

from .spectra import SimpleSpectrum

__all__ = ["instr_model", "combine_spectra", "match_spectra"]

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


def instr_model(observed_spectrum, model_phot, model_spot, log_spotted_area,
                res, *lam_offsets, err_bar=err_bar):
                # lam_offset, res, err_bar=err_bar):

    # Kernel for instrumental/rotational broadening profile:
    kernel_0 = gaussian_kernel(50, res)
    spotted_area = np.exp(log_spotted_area)

    # observed_spectrum = deepcopy(observed_spectrum)
    # observed_spectrum.convolve(kernel=kernel_0)

    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
    combined_spectrum.convolve(kernel=kernel_0)

    # Apply wavelength correction just to red wavelengths:
    corrected_wavelengths = observed_spectrum.wavelength.copy()

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

        a = combined_interp[i_min:i_max, np.newaxis]
        b = observed_spectrum.flux[i_min:i_max]
        c = np.linalg.inv(a.T @ a) @ a.T @ b  # Ordinary least squares

        residuals += np.sum((a @ c - b)**2)

        combined_scaled[i_min:i_max] = combined_interp[i_min:i_max] * c#[0]

   # residuals /= np.exp(lnf)**2 #err_bar**2

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

#  def instr_model(temp_phot, temp_spot, spotted_area, lam_offset, res,
#                 observed_spectrum, model_grid, err_bar=err_bar):
#
#     # Kernel for instrumental broadening profile:
#     kernel = gaussian(int(5*res), res)
#
#     # from .spectra import slice_spectrum
#     lam_min = observed_spectrum.wavelength.min().value
#     lam_max = observed_spectrum.wavelength.max().value
#
# #    model_phot = model_grid.nearest_spectrum(temp_phot)
# #    model_phot.slice(lam_min, lam_max)
# #    model_spot = model_grid.nearest_spectrum(temp_spot)
# #    model_spot.slice(lam_min, lam_max)
#
#     model_phot = model_grid.spectrum(temp_phot)
#     #model_phot.slice(lam_min, lam_max)
#     model_spot = model_grid.spectrum(temp_spot)
#     #model_spot.slice(lam_min, lam_max)
#
#     combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
#     combined_spectrum.convolve(kernel=kernel)
#
#     # Apply wavelength correction just to red wavelengths:
#     corrected_wavelengths = observed_spectrum.wavelength.copy()
#     mid_wavelengths = (corrected_wavelengths > 7000*u.Angstrom) & (corrected_wavelengths < 8500*u.Angstrom)
# #    blue_wavelengths = np.logical_not(red_wavelengths)
#     blue_wavelengths = (corrected_wavelengths < 7000*u.Angstrom)
#     red_wavelengths = corrected_wavelengths > 8500*u.Angstrom
#     corrected_wavelengths[mid_wavelengths] -= lam_offset*u.Angstrom
#     corrected_wavelengths[blue_wavelengths] -= (lam_offset + 0.35)*u.Angstrom
#     corrected_wavelengths[red_wavelengths] -= (lam_offset - 0.35)*u.Angstrom
#
#     combined_interp = combined_spectrum.interpolate(corrected_wavelengths)
#     # combined_interp = combined_spectrum.interpolate(observed_spectrum.wavelength -
#     #                                                 lam_offset*u.Angstrom)
#
#     combined_scaled = combined_interp.copy()
#     residuals = 0
#     for i_min, i_max in observed_spectrum.wavelength_splits:
#         c, residuals_i = np.linalg.lstsq(combined_interp[i_min:i_max, np.newaxis],
#                                          observed_spectrum.flux[i_min:i_max, np.newaxis])[0:2]
#         residuals += residuals_i
#         combined_scaled[i_min:i_max] = combined_interp[i_min:i_max] * c[0]
#
#     residuals /= err_bar**2
#
#     return combined_scaled, residuals


# def combine_spectra(spectrum_phot, spectrum_spot, spotted_area):
#     combined_flux = (spectrum_phot.flux * (1 - spotted_area) +
#                      spectrum_spot.flux * spotted_area)
#     return SimpleSpectrum(spectrum_phot.wavelength, combined_flux,
#                           dispersion_unit=spectrum_phot.dispersion_unit)


def combine_spectra(spectrum_phot, spectrum_spot, spotted_area):
    spectrum_spot_interp = np.interp(spectrum_phot.wavelength.value,
                                     spectrum_spot.wavelength.value,
                                     spectrum_spot.flux)
    combined_flux = (spectrum_phot.flux * (1 - spotted_area) +
                     spectrum_spot_interp * spotted_area)
    return SimpleSpectrum(spectrum_phot.wavelength, combined_flux,
                          dispersion_unit=spectrum_phot.dispersion_unit)

