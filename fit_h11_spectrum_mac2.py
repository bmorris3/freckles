import time
start = time.time()
import sys
import os
from glob import glob
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import emcee
from emcee.utils import MPIPool
from sklearn import linear_model
from scipy.signal import gaussian

from toolkit import (get_phoenix_model_spectrum, SimpleSpectrum,
                     slice_spectrum, concatenate_spectra, instr_model,
                     combine_spectra, bands_balmer, PHOENIXModelGrid)

model_grid = PHOENIXModelGrid() #ModelGrid()
fixed_temp_phot = 4780
fixed_temp_spot = fixed_temp_phot - 300
model_phot = model_grid.spectrum(fixed_temp_phot, 4.5, 0)
model_spot = model_grid.spectrum(fixed_temp_spot, 4.5, 0)


file_index = sys.argv[1]

target_spectrum = SimpleSpectrum.from_hdf5("HATP11", 0)

spec_band = []
for band in bands_balmer:
    target_slice = slice_spectrum(target_spectrum, band.min, band.max)
    target_slice.flux /= target_slice.flux.max()
    spec_band.append(target_slice)
slices = concatenate_spectra(spec_band)


def instr_model(teff, logg, z, lam_offset, res, observed_spectrum):
    kernel = gaussian(int(5*res), res)

    # Apply wavelength correction just to red wavelengths:
    corrected_wavelengths = observed_spectrum.wavelength.copy()
    corrected_wavelengths -= lam_offset*u.Angstrom

    combined_spectrum = model_grid.spectrum(teff, logg, z,
                                            wavelengths=corrected_wavelengths.value)
    combined_spectrum.convolve(kernel=kernel)

    #combined_interp = combined_spectrum.interpolate(corrected_wavelengths)

    A = np.vstack([combined_spectrum.flux, corrected_wavelengths.value]).T
    
    combined_scaled = combined_spectrum.flux.copy()
    residuals = 0
    for i_min, i_max in observed_spectrum.wavelength_splits:

        c, residuals_i = np.linalg.lstsq(A[i_min:i_max, :], 
                                         observed_spectrum.flux[i_min:i_max, np.newaxis])[0:2]
    
        residuals += residuals_i
        combined_scaled[i_min:i_max] = (c[0] * combined_spectrum.flux[i_min:i_max] +
                                        c[1] * corrected_wavelengths[i_min:i_max].value)

    return combined_scaled, residuals


def lnprior(theta):
    teff, logg, z, dlam, lnf, res = theta
    if ((4000 < teff < 5200) and (4 < logg < 5) and (-0.5 < z < 0.5) and
        (-3 < dlam < 3) and (-10 < lnf < -1) and (0.5 < res < 3)):
        return 0.0
    return -np.inf

yerr = 0.01


def lnlike(theta):
    teff, logg, z, dlam, lnf, res = theta
    model, residuals = instr_model(teff, logg, z, dlam, res, slices)
    # return -0.5*residuals

    # Source: http://dan.iel.fm/emcee/current/user/line/#maximum-likelihood-estimation
    inv_sigma2 = 1.0 / (yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*np.sum((model - slices.flux)**2 * inv_sigma2 - np.log(inv_sigma2))


def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

ndim, nwalkers = 6, 12
pos = []

while len(pos) < nwalkers: 
    try_this = (np.array([4780, 4.6, 0.3, 0, -6, 1.8]) +
                np.array([500, 0.2, 0.5, 0.001, 0.05, 0.5]) * np.random.randn(ndim))
    if np.isfinite(lnlike(try_this)):
        pos.append(try_this)

# pool = MPIPool(loadbalance=True)
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)#, pool=pool)

print("Running MCMC burn-in...")
pos1 = sampler.run_mcmc(pos, 1000)[0]#, rstate0=np.random.get_state())

print("Running MCMC...")
sampler.reset()
#pos2 = sampler.run_mcmc(pos1, 20000)[0]
pos2 = sampler.run_mcmc(pos1, 5000)[0]
end = time.time()
print("runtime", (end-start)/60)
print("MCMC done")

#pool.close()
outfile_path = sys.argv[2]
output_path = os.path.abspath(os.path.join(outfile_path, 'chains_{0:02d}.txt'.format(int(file_index))))
lnprob_path = os.path.abspath(os.path.join(outfile_path, 'lnprob_{0:02d}.txt'.format(int(file_index))))
np.savetxt(output_path, sampler.flatchain[-10:, :])
np.savetxt(lnprob_path, sampler.flatlnprobability)
#np.save('lastthousand.txt', sampler.flatchain[-1000:, :])
