# -*- coding: utf-8 -*-
#
#  losses.py
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

import torch


class Loss:

    def __init__(self, criterion: str = 'l2', eps: float = 0.001, nu: float = 0.1):
        self.criterion: str = criterion
        assert self.criterion in {'l1', 'l2', 'nu-svr', 'eps-svr'}
        self.eps: float = eps
        self.nu: float = nu

    def __call__(self, residuals: torch.Tensor) -> torch.Tensor:
        if self.criterion == 'l1':
            return torch.abs(residuals)
        elif self.criterion == 'l2':
            return torch.square(residuals)
        elif self.criterion == 'nu-svr':
            quantiles = torch.quantile(residuals, self.nu, dim=0).unsqueeze(0)
            weights = (residuals > quantiles)
            return weights * quantiles
        elif self.criterion == 'eps-svr':
            return torch.clamp(torch.abs(residuals) - self.eps, min=0)
        else:
            raise NotImplementedError(f'Unknown criterion "{self.criterion}"')
