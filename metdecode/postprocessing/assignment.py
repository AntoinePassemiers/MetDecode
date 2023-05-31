# -*- coding: utf-8 -*-
#
#  assignment.py
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
import torch


def cell_type_assignment(R_reference: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    m = len(R_reference)
    rho = np.abs(np.corrcoef(np.concatenate((R_reference.cpu().data.numpy(), R.cpu().data.numpy()), axis=0)))
    rho = rho[:m, m:]
    idx = torch.zeros(len(R), dtype=torch.long)
    for _ in range(len(idx)):
        i, j = np.unravel_index(np.argmax(rho), rho.shape)
        rho[i, :] = -1
        rho[:, j] = -1
        idx[j] = i
    return R[idx, :]
