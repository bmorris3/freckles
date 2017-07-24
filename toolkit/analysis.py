from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from .spectra import SimpleSpectrum

__all__ = ["instr_model", "combine_spectra"]


def instr_model(temp_phot, temp_spot, spotted_area, lam_offset,
                observed_spectrum, model_grid):

    # from .spectra import slice_spectrum
    lam_min = observed_spectrum.wavelength.min().value
    lam_max = observed_spectrum.wavelength.max().value

    # model_phot = slice_spectrum(model_grid.spectrum(temp_phot),
    #                             lam_min, lam_max)
    # model_spot = slice_spectrum(model_grid.spectrum(temp_spot),
    #                             lam_min, lam_max)

    model_phot = model_grid.spectrum(temp_phot)
    model_phot.slice(lam_min, lam_max)
    model_spot = model_grid.spectrum(temp_spot)
    model_spot.slice(lam_min, lam_max)

    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
    combined_spectrum.convolve()
    combined_interp = combined_spectrum.interpolate(observed_spectrum.wavelength -
                                                    lam_offset*u.Angstrom)

    c, residuals = np.linalg.lstsq(combined_interp[:, np.newaxis],
                                   observed_spectrum.flux[:, np.newaxis])[0:2]

    # from scipy.linalg import lstsq
    # c, residuals = lstsq(combined_interp[:, np.newaxis],
    #                      observed_spectrum.flux[:, np.newaxis],
    #                      check_finite=False, lapack_driver='gelsy')[0:2]

    return combined_interp * c[0], residuals


def combine_spectra(spectrum_phot, spectrum_spot, spotted_area):
    combined_flux = (spectrum_phot.flux * (1 - spotted_area) +
                     spectrum_spot.flux * spotted_area)
    return SimpleSpectrum(spectrum_phot.wavelength, combined_flux,
                          dispersion_unit=spectrum_phot.dispersion_unit)
