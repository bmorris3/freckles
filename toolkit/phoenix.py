from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import h5py
import numpy as np
from astropy.utils.data import download_file
from astropy.io import fits
import astropy.units as u
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

__all__ = ['get_phoenix_model_spectrum', 'phoenix_model_temps',
           'ModelGrid']

phoenix_model_temps = np.array(
    [1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2050, 2100,
     2150, 2200, 2250, 2300, 2350, 2400, 2500, 2600, 2700, 2800, 2900,
     3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
     4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100,
     5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200,
     6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000])


model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,
                                          'data', 'model_grid.npz'))

btsettl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,
                                            'data', 'lte{T_eff:03d}.0-4.5-0.0a+'
                                                    '0.0.BT-Settl.spec.fits'))

phoenix_grid_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 os.pardir, 'data',
                                                 'phoenix_grid.hdf5'))


def get_path(T_eff):
    closest_ind = np.argmin(np.abs(phoenix_model_temps - T_eff))
    closest_grid_temperature = phoenix_model_temps[closest_ind]
    path = btsettl_path.format(T_eff=closest_grid_temperature//100)
    return path


def nrefrac(wavelength, density=1.0):
    """Calculate refractive index of air from Cauchy formula.

    Input: wavelength in Angstrom, density of air in amagat (relative to STP,
    e.g. ~10% decrease per 1000m above sea level).
    Returns N = (n-1) * 1.e6. 
    """

    # The IAU standard for conversion from air to vacuum wavelengths is given
    # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
    # Angstroms, convert to air wavelength (AIR) via: 

    #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

    wl = np.array(wavelength)

    wl2inv = (1.0e4/wl)**2
    refracstp = 272.643 + 1.2288 * wl2inv  + 3.555e-2 * wl2inv**2
    return density * refracstp


def get_phoenix_model_spectrum(T_eff, log_g=4.5, cache=True):
    """
    Download a PHOENIX model atmosphere spectrum for a star with given
    properties.

    Parameters
    ----------
    T_eff : float
        Effective temperature. The nearest grid-temperature will be selected.
    log_g : float
        This must be a log g included in the grid for the effective temperature
        nearest ``T_eff``.
    cache : bool
        Cache the result to the local astropy cache. Default is `True`.

    Returns
    -------
    spectrum : `~specutils.Spectrum1D`
        Model spectrum
    """
    fluxes_path = get_path(T_eff=T_eff)
    data = fits.getdata(fluxes_path)
    fluxes = data['Flux']
    wavelengths_vacuum = data['Wavelength']

    # Wavelengths are provided at vacuum wavelengths. For ground-based
    # observations convert this to wavelengths in air, as described in
    # Husser 2013, Eqns. 8-10:
    # sigma_2 = (10**4 / wavelengths_vacuum)**2
    # f = (1.0 + 0.05792105/(238.0185 - sigma_2) + 0.00167917 /
         # (57.362 - sigma_2))
    # wavelengths_air = wavelengths_vacuum / f

    # from the manual at https://phoenix.ens-lyon.fr/Grids/FORMAT
    VAC = wavelengths_vacuum * 1e4
    #AIR = VAC / (1.0 + 2.735182e-4 + 131.4182 / VAC**2 + 2.76249e8 / VAC**4)
    wavelengths_air = VAC / (1.0 + 2.735182e-4 + 131.4182 / VAC**2 + 2.76249e8 / VAC**4)
    
    from .spectra import SimpleSpectrum  # Prevent circular imports

    spectrum = SimpleSpectrum(wavelengths_air * u.Angstrom, fluxes,
                              dispersion_unit=u.Angstrom)

    return spectrum


def construct_model_grid(temp_min=3000, temp_max=6000):

    tmp_model = get_phoenix_model_spectrum(4700)
    test_temps = np.sort(phoenix_model_temps[(phoenix_model_temps <= temp_max) &
                                             (phoenix_model_temps >= temp_min)])
    all_models = np.zeros((tmp_model.flux.shape[0], len(test_temps)))
    wavelengths = np.zeros(tmp_model.flux.shape[0])

    for i, test_temp in enumerate(test_temps):
        if i == 0:
            wavelengths_order = np.argsort(tmp_model.wavelength)
            wavelengths = tmp_model.wavelength[wavelengths_order]

        each_model = get_phoenix_model_spectrum(test_temp, log_g=4.5,
                                                cache=True)
        all_models[:, i] = each_model.flux[wavelengths_order]
    return wavelengths, test_temps, all_models


model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir,
                                          'data', 'model_grid.npz'))


class ModelGrid(object):
    def __init__(self, path=model_path, temp_min=3000, temp_max=6000,
                 spline_order=1):
        if os.path.exists(path):
            pickled_grid = np.load(path)

            self.wavelengths = pickled_grid['wavelengths']
            self.test_temps = pickled_grid['test_temps']
            self.all_models = pickled_grid['all_models']

        else:
            wavelengths, test_temps, all_models = construct_model_grid(temp_min,
                                                                       temp_max)
            np.savez(path, wavelengths=wavelengths.value, test_temps=test_temps,
                     all_models=all_models)

            self.wavelengths = wavelengths
            self.test_temps = test_temps
            self.all_models = all_models

        self._interp = None
        self.spline_order = spline_order
        self.cache = dict()

    def interp(self, lam, temp):

        if self._interp is None:
            self._interp = RectBivariateSpline(self.wavelengths,
                                               self.test_temps,
                                               self.all_models,
                                               kx=self.spline_order,
                                               ky=self.spline_order)

        # if temp not in self.cache:
        #     if len(self.cache) > 0:
        #         cached_temps = np.array(list(self.cache))
        #         nearest_temp = cached_temps[np.argmin(np.abs(cached_temps - temp))]
        #
        #         if abs(temp - nearest_temp) < 1:
        #             return self.cache[nearest_temp]
        #
        #     self.cache[temp] = self._interp(lam, temp)
        # return self.cache[temp]
        return self._interp(lam, temp)

    def interp_reshape(self, lam, temp):
        return self.interp(lam, temp)[:, 0]

    def spectrum(self, temp):
        """
        Get a full resolution PHOENIX model spectrum interpolated from
        the grid at temperature ``temp``
        """
        from .spectra import SimpleSpectrum

        if temp in phoenix_model_temps:
            this_temperature = temp == self.test_temps
            flux = np.compress(this_temperature, self.all_models, axis=1)[:, 0]

        else:
            flux = self.interp_reshape(self.wavelengths, temp)

        #flux /= flux.max()

        return SimpleSpectrum(self.wavelengths, flux,
                              dispersion_unit=u.Angstrom)

    def nearest_spectrum(self, temp, to_nearest=25):
        """
        Get a full resolution PHOENIX model spectrum interpolated from
        the grid at temperature ``temp``
        """
        from .spectra import SimpleSpectrum

        if temp in phoenix_model_temps:
            this_temperature = temp == self.test_temps
            flux = np.compress(this_temperature, self.all_models, axis=1)[:, 0]
        else:
            rounded_temperature = round(temp / to_nearest) * to_nearest
            flux = self.interp_reshape(self.wavelengths, rounded_temperature)

        flux /= flux.max()

        return SimpleSpectrum(self.wavelengths, flux,
                              dispersion_unit=u.Angstrom)


wavelength_min = 3000  # Angstrom
wavelength_max = 10000  # Angstrom

phoenix_model_temps = np.array(
    [2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
     3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
     4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
     5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800,
     5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700,
     6800, 6900, 7000, 7200, 7400, 7600, 7800, 8000, 8200,
     8400, 8600, 8800, 9000, 9200, 9400, 9600, 9800, 10000,
     10200, 10400, 10600, 10800, 11000, 11200, 11400, 11600, 11800,
     12000, 12500, 13000, 13500, 14000, 14500, 15000])

phoenix_model_metallicities = np.array([-4, -3, -2, -1.5, -1, -0.5, -0, 0.5,
                                        1.0])

phoenix_model_gravities = np.arange(0, 6.5, 0.5)


def get_any_metallicity_url(T_eff, log_g, z):
    closest_grid_temperature = phoenix_model_temps[np.argmin(np.abs(phoenix_model_temps - T_eff))]

    if z > 0:
        z = "+{0:1.1f}".format(z)
    elif z == 0:
        z = "-{0:1.1f}".format(z)
    else:
        z = "{0:1.1f}".format(z)

    url = ('ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/'
           'PHOENIX-ACES-AGSS-COND-2011/Z{z}/lte{T_eff:05d}-'
           '{log_g:1.2f}{z}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
           ).format(T_eff=closest_grid_temperature, log_g=log_g, z=z)
    return url


def get_phoenix_model_wavelengths(cache=True):
    """
    Return the wavelength grid that the PHOENIX models were computed on,
    transformed into wavelength units in air (not vacuum).
    """
    wavelength_url = ('ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/'
                      'HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    wavelength_path = download_file(wavelength_url, cache=cache, timeout=30)
    wavelengths_vacuum = fits.getdata(wavelength_path)

    # Wavelengths are provided at vacuum wavelengths. For ground-based
    # observations convert this to wavelengths in air, as described in
    # Husser 2013, Eqns. 8-10:
    sigma_2 = (10**4 / wavelengths_vacuum)**2
    f = (1.0 + 0.05792105/(238.0185 - sigma_2) + 0.00167917 /
         (57.362 - sigma_2))
    wavelengths_air = wavelengths_vacuum / f
    return wavelengths_air


class PHOENIXModelGrid(object):
    def __init__(self, path=phoenix_grid_path,
                 interp_wavelength_min=6562.8-5,
                 interp_wavelength_max=6562.8+5):
        self.path = path

        if not os.path.exists(path):
            raise ValueError('No such HDF5 archive {0}'.format(path))

        archive = h5py.File(path, 'r')
        dset = archive['spectra']
        self.temperatures = dset.attrs['temperatures']
        self.gravities = dset.attrs['gravities']
        self.metallicities = dset.attrs['metallicities']

        all_wavelengths = get_phoenix_model_wavelengths()
        wavelength_mask = ((all_wavelengths < wavelength_max) &
                           (all_wavelengths > wavelength_min))
        wavelengths_in_bounds = all_wavelengths[wavelength_mask]
        interp_bounds = ((wavelengths_in_bounds < interp_wavelength_max) &
                         (wavelengths_in_bounds > interp_wavelength_min))
        self.wavelengths = wavelengths_in_bounds[interp_bounds]


        points = (self.wavelengths, self.temperatures, self.gravities,
                  self.metallicities)
        values = dset[np.where(interp_bounds)[0], :, :, :][:]

        rgi = RegularGridInterpolator(points, values)
        self._rgi = rgi

    def interp(self, temperature, gravity, metallicity, wavelengths=None,
               method='linear'):
        if wavelengths is None:
            wavelengths = self.wavelengths
        xi = np.hstack([wavelengths[:, np.newaxis],
                        np.repeat([[temperature, gravity, metallicity]],
                                  len(wavelengths), axis=0)])
        return self._rgi(xi, method=method)

    @classmethod
    def download_phoenix_spectrum_grid(cls, teff_min, teff_max, log_g_min,
                                       log_g_max, z_min, z_max,
                                       wavelength_min=wavelength_min,
                                       wavelength_max=wavelength_max,
                                       path=phoenix_grid_path):
        """
        Get a grid of PHOENIX model spectra.

        Parameters
        ----------
        teff_min : float
            Minimum effective temperature (inclusive)
        teff_max : float
            Maximum effective temperature (inclusive)
        log_g_min : float
            Minimum surface gravity (inclusive)
        log_g_max : float
            Maximum surface gravity (inclusive)
        z_min : float
            Minimum metallicity (inclusive)
        z_max : float
           Maximum metallicity (inclusive)
        wavelength_min : float
            Minimum wavelength to save (angstrom)
        wavelength_max : float
            Maximum wavelength to save (angstrom)

        Notes
        -----
        The HDF5 archive that gets saved is roughly 0.5 GB.
        """
        temps = ((phoenix_model_temps <= teff_max) &
                 (phoenix_model_temps >= teff_min))
        gravs = ((phoenix_model_gravities <= log_g_max) &
                 (phoenix_model_gravities >= log_g_min))
        metals = ((phoenix_model_metallicities <= z_max) &
                  (phoenix_model_metallicities >= z_min))

        wavelengths = get_phoenix_model_wavelengths()

        wavelength_mask = ((wavelengths < wavelength_max) &
                           (wavelengths > wavelength_min))
        wavelengths_in_bounds = wavelengths[wavelength_mask]

        archive = h5py.File(path, 'w')

        data_cube_shape = (len(wavelengths_in_bounds), np.count_nonzero(temps),
                           np.count_nonzero(gravs), np.count_nonzero(metals))
        dset = archive.create_dataset('spectra', shape=data_cube_shape,
                                      dtype=np.float32, compression='gzip')

        dset.attrs['temperatures'] = phoenix_model_temps[temps]
        dset.attrs['gravities'] = phoenix_model_gravities[gravs]
        dset.attrs['metallicities'] = phoenix_model_metallicities[metals]

        for i, t in enumerate(phoenix_model_temps[temps]):
            for j, g in enumerate(phoenix_model_gravities[gravs]):
                for k, z in enumerate(phoenix_model_metallicities[metals]):
                    if np.all(dset[:, i, j, k] == 0):
                        url = get_any_metallicity_url(t, g, z)
                        tmp_path = download_file(url, cache=False, timeout=30)
                        spectrum = fits.getdata(tmp_path)[wavelength_mask]
                        dset[:, i, j, k] = spectrum
                archive.flush()

        archive.close()
        return cls(path=path)

    def spectrum(self, temperature, gravity, metallicity, wavelengths=None,
                 method='linear'):
        if wavelengths is None:
            wavelengths = self.wavelengths
        flux = self.interp(temperature, gravity, metallicity, method=method,
                           wavelengths=wavelengths)
        from .spectra import Spectrum1D
        return Spectrum1D(wavelengths if hasattr(wavelengths, 'unit') else
                          u.Quantity(wavelengths, u.Angstrom), flux)
