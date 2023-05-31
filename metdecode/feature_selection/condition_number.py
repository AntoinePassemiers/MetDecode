# -*- coding: utf-8 -*-
#
#  condition_number.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import numpy as np


def condition_number_based_fs(
        R_atlas: np.ndarray,
        R_cfdna: np.ndarray,
        fraction: float = 0.9
) -> np.ndarray:

    n_features = int(np.clip(round(fraction * R_cfdna.shape[1]), 0, R_cfdna.shape[1]))
    inv = np.linalg.pinv(R_atlas.T)
    print(inv.shape, R_cfdna.shape)
    cn = np.outer(np.linalg.norm(inv, axis=1), np.linalg.norm(R_cfdna, axis=1))
    cn /= np.abs(np.dot(inv.T, R_cfdna))
    print(cn.shape)
    cn = np.max(cn, axis=0)
    assert False
