import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import astropy.units as u
import os
from itertools import combinations
import emcee
from emcee.utils import MPIPool

from toolkit import (get_phoenix_model_spectrum, EchelleSpectrum, ModelGrid,
                     slice_spectrum, concatenate_spectra, bands_TiO, instr_model, 
                     combine_spectra)
from scipy.signal import gaussian

model_grid = ModelGrid()
fixed_temp_phot = 4780
fixed_temp_spot = fixed_temp_phot - 300
model_phot = model_grid.spectrum(fixed_temp_phot)
model_spot = model_grid.spectrum(fixed_temp_spot)


fits_files = []
home_dir = '/usr/lusers/bmmorris/freckles_data/'


#for dirpath, dirnames, files in os.walk('/local/tmp/freckles/data/'):
for dirpath, dirnames, files in os.walk(home_dir):
    for file in files:
        file_path = os.path.join(dirpath, file)
        if (file_path.endswith('.fits') and ('weird' not in file_path) 
             and ('dark' not in file_path) and ('HAT' in file_path)):
            fits_files.append(file_path)

fits_files = fits_files[1:]

new_paths = []
for path in fits_files: 
    split_name = path.split(os.sep)
    date = split_name[-2]
    fname = split_name[-1].split('.')
    new_paths.append('fits/' + '.'.join([date] + fname[:2]) + '.npy')
    
def plot_spliced_spectrum(observed_spectrum, model_flux, other_model=None):
    n_chunks = len(slices.wavelength_splits)
    fig, ax = plt.subplots(n_chunks, 1, figsize=(8, 10))

    for i, inds in enumerate(observed_spectrum.wavelength_splits):
        min_ind, max_ind = inds

        ax[i].errorbar(observed_spectrum.wavelength[min_ind:max_ind].value, 
                       observed_spectrum.flux[min_ind:max_ind], 
                       0.025*np.ones(max_ind-min_ind))
        ax[i].plot(observed_spectrum.wavelength[min_ind:max_ind], 
                   model_flux[min_ind:max_ind])
        
        if other_model is not None:
            ax[i].plot(observed_spectrum.wavelength[min_ind:max_ind], 
                       other_model[min_ind:max_ind], alpha=0.4)
        
        ax[i].set_xlim([observed_spectrum.wavelength[max_ind-1].value, 
                        observed_spectrum.wavelength[min_ind].value])
        ax[i].set_ylim([0.9*observed_spectrum.flux[min_ind:max_ind].min(), 
                        1.1])

    return fig, ax

def nearest_order(spectrum, wavelength):
    return np.argmin([abs(spec.wavelength.mean() - wavelength).value
                      for spec in spectrum.spectrum_list])

def nearest_order(spectrum, wavelength):
    return np.argmin([abs(spec.wavelength.mean() - wavelength).value
                      for spec in spectrum.spectrum_list])

#home_dir = '/local/tmp/freckles/' if os.uname().sysname == 'Linux' else os.path.expanduser('~')
standard_path = os.path.join(home_dir, 'Q3UW04/UT160706/BD28_4211.0034.wfrmcpc.fits')
#standard_path = os.path.join(home_dir, 'data/Q3UW04/UT160706/BD28_4211.0034.wfrmcpc.fits')

standard_spectrum = EchelleSpectrum.from_fits(standard_path)

import sys
file_index = sys.argv[1]
in_path = fits_files[int(file_index)]
#in_path = os.path.join('/run/media/bmmorris/PASSPORT/APO/Q3UW04/UT160703',
#                       'KIC9652680.0028.wfrmcpc.fits')
                        #'KIC9652680.0025.wfrmcpc.fits')#fits_files[-2]
print(in_path)
target_spectrum = EchelleSpectrum.from_fits(in_path)
only_orders = list(range(len(target_spectrum.spectrum_list)))
target_spectrum.continuum_normalize(standard_spectrum,
                                    polynomial_order=10,
                                    only_orders=only_orders,
                                    plot_masking=False)

rv_shifts = u.Quantity([target_spectrum.rv_wavelength_shift(order, T_eff=5618)
                        for order in only_orders])
median_rv_shift = np.median(rv_shifts)

target_spectrum.offset_wavelength_solution(median_rv_shift)

spec_band = []
for band in bands_TiO:
    band_order = target_spectrum.get_order(nearest_order(target_spectrum, band.core))
    target_slice = slice_spectrum(band_order, band.min, band.max)
    target_slice.flux /= target_slice.flux.max()
    spec_band.append(target_slice)
slices = concatenate_spectra(spec_band)

def instr_model_fixed(spotted_area, lam_offset, res, observed_spectrum):
    
    kernel = gaussian(int(5*res), res)
    combined_spectrum = combine_spectra(model_phot, model_spot, spotted_area)
    combined_spectrum.convolve(kernel=kernel)

    # Apply wavelength correction just to red wavelengths:
    corrected_wavelengths = observed_spectrum.wavelength.copy()
    mid_wavelengths = ((corrected_wavelengths > 7000*u.Angstrom) & 
                       (corrected_wavelengths < 8500*u.Angstrom))
    blue_wavelengths = (corrected_wavelengths < 7000*u.Angstrom)
    red_wavelengths = corrected_wavelengths > 8500*u.Angstrom
    corrected_wavelengths[mid_wavelengths] -= lam_offset*u.Angstrom
    corrected_wavelengths[blue_wavelengths] -= (lam_offset + 0.35)*u.Angstrom
    corrected_wavelengths[red_wavelengths] -= (lam_offset - 0.35)*u.Angstrom

    combined_interp = combined_spectrum.interpolate(corrected_wavelengths)

    A = np.vstack([combined_interp, corrected_wavelengths.value]).T
    
    combined_scaled = combined_interp.copy()
    residuals = 0
    for i_min, i_max in observed_spectrum.wavelength_splits:

        c, residuals_i = np.linalg.lstsq(A[i_min:i_max, :], 
                                         observed_spectrum.flux[i_min:i_max, np.newaxis])[0:2]
    
        residuals += residuals_i
        #combined_scaled[i_min:i_max] = c[0] * combined_interp[i_min:i_max] 
        combined_scaled[i_min:i_max] = (c[0] * combined_interp[i_min:i_max] + 
                                        c[1] * corrected_wavelengths[i_min:i_max].value)

    return combined_scaled, residuals


def lnprior(theta):
    lna, dlam, lnf, res = theta
    if ((-10 < lna < np.log(0.5)) and (-20 < dlam < 20) and 
        (0.5 < res < 2) and (-4 < lnf < -1)):
        return 0.0
    return -np.inf

yerr = 0.01

def lnlike(theta):
    lna, dlam, lnf, res = theta
    model, residuals = instr_model_fixed(np.exp(lna), dlam, res, slices)
    # return -0.5*residuals

    # Source: http://dan.iel.fm/emcee/current/user/line/#maximum-likelihood-estimation
    inv_sigma2 = 1.0 / (yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*np.sum((model - slices.flux)**2 * inv_sigma2 - np.log(inv_sigma2))

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

ndim, nwalkers = 4, 8
pos = []

while len(pos) < nwalkers: 
    try_this = (np.array([np.log(0.3), -1.61, -2.8, 1.14]) + 
                np.array([2, 0.1, 0.2, 0.1]) * np.random.randn(ndim))
    if np.isfinite(lnlike(try_this)):
        pos.append(try_this)

pool = MPIPool(loadbalance=True)
if not pool.is_master():
    pool.wait()
    sys.exit(0)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

print("Running MCMC burn-in...")
pos1 = sampler.run_mcmc(pos, 50)[0]#, rstate0=np.random.get_state())

print("Running MCMC...")
sampler.reset()
pos2 = sampler.run_mcmc(pos1, 500)[0]
print("MCMC done")

pool.close()
outfile_path = sys.argv[2]
output_path = os.path.join(outfile_path, 'chains_{02d}.txt'.format(int(file_index)))
np.savetxt(output_path, sampler.flatchain[-10000:, :])
#np.save('lastthousand.txt', sampler.flatchain[-1000:, :])