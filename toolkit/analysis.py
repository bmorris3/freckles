from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from scipy.signal import gaussian

from .spectra import SimpleSpectrum

__all__ = ["instr_model", "combine_spectra"]


def instr_model(temp_phot, temp_spot, spotted_area, lam_offset, res,
                observed_spectrum, model_grid):

    # Kernel for instrumental broadening profile:
    kernel = gaussian(int(5*res), res)

    # from .spectra import slice_spectrum
    lam_min = observed_spectrum.wavelength.min().value
    lam_max = observed_spectrum.wavelength.max().value

    model_phot = model_grid.spectrum(temp_phot)
    model_phot.slice(lam_min, lam_max)
    model_spot = model_grid.spectrum(temp_spot)
    model_spot.slice(lam_min, lam_max)

    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
    combined_spectrum.convolve(kernel=kernel)
    combined_interp = combined_spectrum.interpolate(observed_spectrum.wavelength -
                                                    lam_offset*u.Angstrom)

    combined_scaled = combined_interp.copy()
    residuals = 0
    for i_min, i_max in observed_spectrum.wavelength_splits:
        c, residuals_i = np.linalg.lstsq(combined_interp[i_min:i_max, np.newaxis],
                                       observed_spectrum.flux[i_min:i_max, np.newaxis])[0:2]
        residuals += residuals_i
        combined_scaled[i_min:i_max] = combined_interp[i_min:i_max] * c[0]

    return combined_scaled, residuals


def combine_spectra(spectrum_phot, spectrum_spot, spotted_area):
    combined_flux = (spectrum_phot.flux * (1 - spotted_area) +
                     spectrum_spot.flux * spotted_area)
    return SimpleSpectrum(spectrum_phot.wavelength, combined_flux,
                          dispersion_unit=spectrum_phot.dispersion_unit)
