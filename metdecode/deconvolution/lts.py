# -*- coding: utf-8 -*-
#
#  lts.py
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

from metdecode.losses import Loss


def least_trimmed_squares(
        criterion: Loss,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        R_cfdna: torch.Tensor,
        weights_cfdna: torch.Tensor,
        n_features: int = 100
) -> torch.Tensor:
    R_reconstructed = torch.mm(alpha, gamma)
    residuals = R_reconstructed - R_cfdna
    idx = torch.argsort(torch.mean(criterion(residuals), dim=0))
    idx = idx[:n_features]

    # Select features
    gamma = gamma[:, idx]
    R_cfdna = R_cfdna[:, idx]
    weights_cfdna = weights_cfdna[:, idx]

    # normalize weights
    weights_cfdna = weights_cfdna / torch.sum(weights_cfdna)

    # Compute LTS loss
    loss = torch.sum(weights_cfdna * criterion(residuals[:, idx]))

    return loss
