# -*- coding: utf-8 -*-
#
#  wasserstein.py
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
import ot
import torch
from scipy.spatial.distance import cdist


def transport_plan(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    assert not np.any(np.isnan(X1))
    assert not np.any(np.isnan(X2))
    distances = cdist(X1, X2)
    assert not np.any(np.isnan(distances ** p))
    assert not np.any(np.isinf(distances ** p))
    return ot.emd(a, b, distances ** p)


def ot_mapping(
        X1: torch.Tensor,
        X2: torch.Tensor,
        p: int = 2
) -> torch.Tensor:
    X1 = X1.cpu().data.numpy()
    X2 = X2.cpu().data.numpy()
    gamma = transport_plan(X1, X2, p=p)
    gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
    return torch.FloatTensor(gamma)
