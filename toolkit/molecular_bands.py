from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.units as u

__all__ = ['Band', 'bands_TiO']


class Band(object):
    """Fitting metadata on molecular absorption bands"""
    def __init__(self, core, min, max):
        self.min = min
        self.max = max
        self.core = core

# Strongest TiO bands from Valenti et al. 1998
# http://adsabs.harvard.edu/abs/1998ApJ...498..851V
strong_lines = u.Quantity([7125.585, 7054.327, 7087.598], u.Angstrom)
band_bounds = u.Quantity([[-2, 5], [-1, 5], [-1, 3], [-1, 0.5]], u.Angstrom)

bands_TiO = [Band(c, c+bounds[0], c+bounds[1])
             for c, bounds in zip(strong_lines, band_bounds)]
