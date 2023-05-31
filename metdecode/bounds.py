# -*- coding: utf-8 -*-
#
#  bounds.py
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

from typing import List

import numpy as np
import torch


class Constraint:

    def __init__(self, j_target: int, js: List[int]):
        self.j_target: int = j_target
        self.js: List[int] = js
        assert len(self.js) >= 1

    def __call__(self, alpha: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        y_pred = alpha[:, self.js[0]]
        for j in self.js[1:]:
            y_pred = y_pred + alpha[:, j]
        ineq = torch.clamp(lb[:, self.j_target] - y_pred, 0, 1)
        ineq = ineq + torch.clamp(y_pred - ub[:, self.j_target], 0, 1)
        return ineq

    def __str__(self) -> str:
        return f'{self.j_target} <- {self.js}'

    def __repr__(self) -> str:
        return self.__str__()


class ConstraintSet:

    def __init__(self, constraints: List[Constraint], lb: np.ndarray, ub: np.ndarray):
        assert np.all(lb <= ub)
        self.constraints: List[Constraint] = constraints
        self.lb: torch.Tensor = torch.FloatTensor(lb)
        self.ub: torch.Tensor = torch.FloatTensor(ub)

    def __call__(self, alpha: torch.Tensor) -> torch.Tensor:
        n_elements = 0
        loss = torch.FloatTensor([0])
        for constraint in self.constraints:
            ineq = constraint(alpha, self.lb, self.ub)
            loss = loss + torch.sum(ineq)
            n_elements += torch.numel(ineq)
        return loss / n_elements

    def __str__(self) -> str:
        return str(self.constraints)
