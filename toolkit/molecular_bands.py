from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.units as u

__all__ = ['Band', 'bands_TiO', 'bands_off_TiO', 'bands_balmer']


class Band(object):
    """Fitting metadata on molecular absorption bands"""
    def __init__(self, core, min, max):
        self.min = min
        self.max = max
        self.core = core

    def __repr__(self):
        unit = u.Angstrom
        return ("<Band: core={0}, (min, max)=({1}, {2}) [{3}]>"
                .format(self.core.to(unit).value, self.min.to(unit).value,
                        self.max.to(unit).value, unit))

# Strongest TiO bands from Valenti et al. 1998
# http://adsabs.harvard.edu/abs/1998ApJ...498..851V
strong_lines = u.Quantity([7054.189, 7087.598, 7125.585, 8859.802], u.Angstrom)
# 5598.410
band_bounds = u.Quantity([[-0.5, 1.5], [-0.5, 1.5], [-0.5, 1.5], [-0.5, 1.5]], u.Angstrom)

bands_TiO = [Band(c, c+bounds[0], c+bounds[1])
             for c, bounds in zip(strong_lines, band_bounds)]

off_band = 5
bands_off_TiO = [Band(c+off_band*u.Angstrom, c+bounds[0]+off_band*u.Angstrom,
                      c+bounds[1]+off_band*u.Angstrom)
                 for c, bounds in zip(strong_lines, band_bounds)]

# https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable2.htm
# removed: 4101.74
#balmer_series = u.Quantity([4340.462, 4861.3615, 6562.8518], u.Angstrom)
#balmer_bounds = u.Quantity([[-1, 1], [-1, 1], [-2, 2]], u.Angstrom)
balmer_series = u.Quantity([4340.462, 4861.3615, 6562.8518, 6562.8518], u.Angstrom)
balmer_bounds = u.Quantity([[-1, 1], [-1, 1], [-2, -0.5], [0.5, 2]], u.Angstrom)
bands_balmer = [Band(c, c+bounds[0], c+bounds[1])
                for c, bounds in zip(balmer_series, balmer_bounds)]
