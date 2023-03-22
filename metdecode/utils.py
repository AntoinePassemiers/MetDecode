# -*- coding: utf-8 -*-
#
#  utils.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import argparse
from typing import Callable

import numpy as np


def bounded_float_type(lb: float = -np.inf, ub: float = np.inf) -> Callable:
    def _bounded_float_type(arg: str):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError('Must be a floating point number')
        if f < lb or f > ub:
            raise argparse.ArgumentTypeError(f'Argument must be in the range [{lb}, {ub}]')
        return f
    return _bounded_float_type
