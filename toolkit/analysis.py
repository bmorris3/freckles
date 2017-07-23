from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy.signal import gaussian
from scipy.ndimage import convolve1d
import astropy.units as u

from .spectra import SimpleSpectrum

__all__ = ['convolve_spectrum', "instr_model", "combine_spectra"]

approx_resolution_ratio = 6.4122026154034
smoothing_kernel = gaussian(int(5*approx_resolution_ratio),
                            approx_resolution_ratio)


def convolve_spectrum(spectrum, kernel=smoothing_kernel):
    convolved_flux = convolve1d(spectrum.flux, kernel)
    convolved_spectrum = SimpleSpectrum(spectrum.wavelength,
                                        convolved_flux/np.median(convolved_flux))
    return convolved_spectrum


def instr_model(temp_phot, temp_spot, spotted_area, lam_offset,
                observed_spectrum, model_grid):

    from .spectra import slice_spectrum, interpolate_spectrum

    model_phot = slice_spectrum(model_grid.spectrum(temp_phot),
                                observed_spectrum.wavelength.min(),
                                observed_spectrum.wavelength.max())
    model_spot = slice_spectrum(model_grid.spectrum(temp_spot),
                                observed_spectrum.wavelength.min(),
                                observed_spectrum.wavelength.max())

    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
    combined_broadened = convolve_spectrum(combined_spectrum)
    combined_interp = interpolate_spectrum(combined_broadened,
                                           observed_spectrum.wavelength - lam_offset*u.Angstrom)

    c, residuals = np.linalg.lstsq(combined_interp.flux[:, np.newaxis],
                                   observed_spectrum.flux[:, np.newaxis])[0:2]
    return combined_interp.flux * c[0], residuals


def combine_spectra(spectrum_phot, spectrum_spot, spotted_area):
    combined_flux = (spectrum_phot.flux * (1 - spotted_area) +
                     spectrum_spot.flux * spotted_area)
    return SimpleSpectrum(spectrum_phot.wavelength, combined_flux)
