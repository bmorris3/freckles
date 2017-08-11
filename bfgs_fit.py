import os
from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.utils.console import ProgressBar

from toolkit import (get_phoenix_model_spectrum, EchelleSpectrum, ModelGrid,
                     slice_spectrum, concatenate_spectra, bands_TiO, instr_model)
from scipy.optimize import fmin_l_bfgs_b
#from astropy.utils.console import ProgressBar

model_grid = ModelGrid()
# Limit combinations such that delta T < 3000 K
# temp_combinations = [i for i in combinations(model_grid.test_temps, 2) 
#                      if (abs(i[0] - i[1]) <= 3000) and (4000 < i[1] < 5000)]

fixed_temp_phot = 4780
temp_combinations = [[i, fixed_temp_phot] for i in model_grid.test_temps
                     if (i < fixed_temp_phot) and (abs(i - fixed_temp_phot) <= 2000)]

n_combinations = len(temp_combinations)
n_fit_params = 3#4
best_parameters = np.zeros((n_combinations, n_fit_params))

fits_files = []

for dirpath, dirnames, files in os.walk('/local/tmp/freckles/data/'):
    for file in files:
        file_path = os.path.join(dirpath, file)
        if (file_path.endswith('.fits') and 'weird' not in file_path and 'dark'
            not in file_path and 'HAT' in file_path):
            fits_files.append(file_path)

fits_files = fits_files[1:]

new_paths = []
for path in fits_files: 
    split_name = path.split(os.sep)
    date = split_name[-2]
    fname = split_name[-1].split('.')
    new_paths.append('fits/' + '.'.join([date] + fname[:2]) + '.npy')

home_dir = '/local/tmp/freckles/' if os.uname().sysname == 'Linux' else os.path.expanduser('~')
standard_path = os.path.join(home_dir, 'data/Q3UW04/UT160706/BD28_4211.0034.wfrmcpc.fits')
standard_spectrum = EchelleSpectrum.from_fits(standard_path)

def nearest_order(spectrum, wavelength):
    return np.argmin([abs(spec.wavelength.mean() - wavelength).value
                      for spec in spectrum.spectrum_list])

with ProgressBar(len(fits_files) * n_combinations) as bar:
    for in_path, out_path in zip(fits_files, new_paths): 
        target_spectrum = EchelleSpectrum.from_fits(in_path)

        only_orders = list(range(len(target_spectrum.spectrum_list)))
        target_spectrum.continuum_normalize(standard_spectrum,
                                            polynomial_order=10,
                                            only_orders=only_orders,
                                            plot_masking=False)

        rv_shifts = u.Quantity([target_spectrum.rv_wavelength_shift(order)
                                for order in only_orders])
        median_rv_shift = np.median(rv_shifts)

        target_spectrum.offset_wavelength_solution(median_rv_shift)

        spec_band = []
        for band in bands_TiO:
            band_order = target_spectrum.get_order(nearest_order(target_spectrum, band.core))
            #target_slice = slice_spectrum(band_order, band.min-5*u.Angstrom, band.max+5*u.Angstrom)
            target_slice = slice_spectrum(band_order, band.min, band.max)
            target_slice.flux /= target_slice.flux.max()
            spec_band.append(target_slice)

        slices = concatenate_spectra(spec_band)
        #slices.plot(normed=False, color='k', lw=2, marker='.')

        def chi2(p, temp_phot, temp_spot):
            #spotted_area, lam_offset, res = p
            ln_spotted_area, lam_offset = p
            spotted_area = np.exp(ln_spotted_area)
            model, residuals = instr_model(temp_phot, temp_spot, spotted_area, 
                                           lam_offset, 9.0, slices, model_grid)
            return residuals

        #bounds = [[0, 0.5], [-10, 10], [5, 15]]
        bounds = [[-10, np.log(0.5)], [-10, 10]]
        #initp = [np.log(0.03), -1.7] # , 9]
        initp = [np.log(0.2), -1.7]

        bfgs_options_fast = dict(epsilon=1e-3, approx_grad=True,
                                 m=10, maxls=20)
        bfgs_options_precise = dict(epsilon=1e-3, approx_grad=True,
                                    m=30, maxls=50)

        for i in range(n_combinations):
            bar.update()
            temp_spot, temp_phot = temp_combinations[i]
            result = fmin_l_bfgs_b(chi2, initp, bounds=bounds, 
                                   args=(temp_phot, temp_spot),
                                   **bfgs_options_precise)
                                   #**bfgs_options_fast)
            best_parameters[i, :] = np.concatenate([result[0], result[1]])
        np.save(out_path, best_parameters)
    #bar.update()

