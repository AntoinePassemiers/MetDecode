# -*- coding: utf-8 -*-
#
#  optimizer.py
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


class Optimizer:

    def __init__(self, alpha_up=1.01, alpha_down=0.9, lr_lb=0, lr_ub=1e+5, verbose=False):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.lr_lb = lr_lb
        self.lr_ub = lr_ub
        self.verbose = verbose

        self.optimizers = []
        self.parameters = []
        self.cursor = 0
        self.last_loss = np.inf

    def add(self, param, lr):
        params = [{
            'params': param,
            'lr': lr
        }]
        self.parameters.append(params[0])
        self.optimizers.append(torch.optim.Adam(params, lr=lr))

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self, loss):

        # Update learning rate
        if self.cursor >= len(self.parameters):
            return
        previous_cursor = (self.cursor + len(self.parameters) - 1) % len(self.parameters)
        if loss <= self.last_loss:
            factor = self.alpha_up
        else:
            factor = self.alpha_down
        lr = self.parameters[previous_cursor]['lr']
        lr = float(np.clip(lr * factor, self.lr_lb, self.lr_ub))
        self.parameters[previous_cursor]['lr'] = lr
        for group in self.optimizers[previous_cursor].param_groups:
            group['lr'] = lr
        self.last_loss = loss

        self.optimizers[self.cursor].step()

        self.cursor = (self.cursor + 1) % len(self.parameters)
