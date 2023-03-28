# -*- coding: utf-8 -*-
#
#  core.py
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

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional
import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import nnls
from scipy.special import logit

from metdecode.optimizer import Optimizer


class BiasCorrection(torch.nn.Module):

    def __init__(
            self,
            n_tissues: int,
            n_markers: int,
            n_unknown_tissues: int = 0,
            multiplicative: bool = False
    ):
        torch.nn.Module.__init__(self)
        self.n_tissues: int = n_tissues
        self.n_hidden: int = 8
        self.n_unknown_tissues: int = n_unknown_tissues
        self.multiplicative: bool = multiplicative

        self.tissue_bias = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=n_tissues)))
        self.marker_bias = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=n_markers)))

        if self.n_unknown_tissues > 0:
            self.unknown_model = torch.nn.Sequential(
                torch.nn.Linear(self.n_tissues, self.n_hidden),
                torch.nn.LayerNorm(self.n_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                torch.nn.LayerNorm(self.n_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                torch.nn.LayerNorm(self.n_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                torch.nn.LayerNorm(self.n_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(self.n_hidden, self.n_unknown_tissues),
            )
        else:
            self.unknown_model = None

    def forward(self, R_logit: torch.Tensor) -> torch.Tensor:

        # Atlas correction
        if self.multiplicative:
            bias_logit = self.marker_bias.unsqueeze(0) * self.tissue_bias.unsqueeze(1)
        else:
            bias_logit = self.marker_bias.unsqueeze(0) + self.tissue_bias.unsqueeze(1)
        R_corrected = torch.sigmoid(R_logit + bias_logit)

        # Modelling of unknown tissues
        if self.unknown_model is not None:
            R_unknown = self.unknown_model.forward(R_corrected.t()).t()
            R_corrected = torch.cat((R_corrected, R_unknown), dim=0)

        return R_corrected


class MetDecode:

    def __init__(
            self,
            max_correction: float = 0.1,
            p: float = 0,
            lambda1: float = 5,
            lambda2: float = 0.02,
            coverage_rcw: bool = True,
            multiplicative: bool = False
    ):
        self.max_correction: float = max_correction
        self.p: float = p
        self.lambda1: float = lambda1
        self.lambda2: float = lambda2
        self.coverage_rcw: bool = coverage_rcw
        self.multiplicative: bool = multiplicative

        self.R_atlas: Optional[np.ndarray] = None
        self.D_atlas: Optional[np.ndarray] = None

    @staticmethod
    def add_pseudo_counts(methylated, depths, pc=0.8):
        methylated = np.copy(methylated)
        depths = np.copy(depths)
        methylated = np.asarray(methylated, dtype=float)
        depths = np.asarray(depths, dtype=float)
        avg_meth = np.sum(methylated) / np.sum(depths)

        # Compute per-row methylation
        row_methylated = np.sum(methylated, axis=1)
        row_depths = np.sum(depths, axis=1)
        mask = np.logical_or(row_methylated == 0, row_methylated == row_depths)
        row_depths[mask] += pc
        row_methylated[mask] += pc * avg_meth
        row_meth = row_methylated / row_depths

        # Compute per-column methylation
        col_methylated = np.sum(methylated, axis=0)
        col_depths = np.sum(depths, axis=0)
        mask = np.logical_or(col_methylated == 0, col_methylated == col_depths)
        col_depths[mask] += pc
        col_methylated[mask] += pc * avg_meth
        col_meth = col_methylated / col_depths

        # Impute missing values
        # TODO: use survival function of Beta distribution instead
        meth_prior = np.outer(np.ones(len(row_meth)), col_meth)
        meth_prior = np.clip(meth_prior, 0.01, 0.99)

        mask = np.logical_or(methylated == 0, methylated == depths)
        methylated[mask] += pc * meth_prior[mask]
        depths[mask] += pc

        assert np.all(methylated > 0)
        assert not np.any(np.isnan(methylated))
        assert not np.any(np.isnan(depths))
        assert not np.any(np.isinf(methylated))
        assert not np.any(np.isinf(depths))
        assert np.all(methylated < depths)

        return methylated, depths

    @staticmethod
    def rcw(mat: np.ndarray) -> np.ndarray:
        return np.log(2 + mat)

    @staticmethod
    def nnls(R_atlas: np.ndarray, R_cfdna: np.ndarray, W_cfdna: np.ndarray, n_unknowns: int) -> np.ndarray:
        n_samples = len(R_cfdna)
        alpha_hat = []
        for i in range(n_samples):
            A = (np.sqrt(W_cfdna[np.newaxis, i, :]) * R_atlas).T
            b = np.sqrt(W_cfdna[i, :]) * R_cfdna[i, :]
            x, residuals = nnls(A, b, maxiter=3000)
            if n_unknowns > 0:
                x = np.concatenate((x, np.full(n_unknowns, 0.2 / n_unknowns)))
            alpha_hat.append(x)
        alpha_hat = np.asarray(alpha_hat)
        alpha_hat /= alpha_hat.sum(axis=1)[:, np.newaxis]
        return alpha_hat

    @staticmethod
    def init_alpha_and_gamma(
            R_atlas: np.ndarray,
            R_cfdna: np.ndarray,
            W_cfdna: np.ndarray,
            n_unknowns: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha_hat = MetDecode.nnls(R_atlas, R_cfdna, W_cfdna, n_unknowns)
        alpha_hat += 1e-6
        alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]
        alpha_logit = np.log(alpha_hat)

        gamma_logit = logit(R_atlas)

        return alpha_logit, gamma_logit

    def compute_weights(self, depths: np.ndarray) -> np.ndarray:
        if self.coverage_rcw:
            depths = MetDecode.rcw(depths)
        depths = depths ** self.p
        return depths / np.sum(depths)

    def fit(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray,
            M_cfdna: np.ndarray,
            D_cfdna: np.ndarray,
            n_unknown_tissues: int = 0,
            max_n_iter: int = 2000,
            patience: int = 100
    ) -> 'MetDecode':

        n_features = D_cfdna.shape[1]
        n_known_tissues = D_atlas.shape[0]
        n_tissues = n_known_tissues + n_unknown_tissues

        # Add pseudo-counts
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)
        M_cfdna, D_cfdna = MetDecode.add_pseudo_counts(M_cfdna, D_cfdna)

        # Compute methylation ratios
        R_atlas = torch.FloatTensor(M_atlas / D_atlas)
        R_cfdna = torch.FloatTensor(M_cfdna / D_cfdna)

        # Compute importance weights based on coverage
        weights_atlas = torch.FloatTensor(self.compute_weights(D_atlas))
        weights_cfdna = torch.FloatTensor(self.compute_weights(D_cfdna))

        # Compute lower and upper bounds on methylation ratios
        mc = self.max_correction
        with torch.no_grad():
            if n_unknown_tissues > 0:
                lb = torch.cat((R_atlas - mc, torch.zeros((n_unknown_tissues, n_features))), dim=0)
                ub = torch.cat((R_atlas + mc, torch.ones((n_unknown_tissues, n_features))), dim=0)
            else:
                lb = R_atlas - mc
                ub = R_atlas + mc

        bias_correction = BiasCorrection(
            n_known_tissues,
            D_cfdna.shape[1],
            n_unknown_tissues=n_unknown_tissues,
            multiplicative=self.multiplicative
        )

        print(f'[MD] Unknown tissues    : {n_unknown_tissues}')
        print(f'[MD] Maximum correction : {self.max_correction}')
        print(f'[MD] p coefficient      : {self.p}')

        info = {'loss': []}

        alpha_logit, gamma_logit = MetDecode.init_alpha_and_gamma(
            R_atlas.cpu().data.numpy(), R_cfdna.cpu().data.numpy(),
            weights_cfdna.cpu().data.numpy(), n_unknown_tissues
        )
        with torch.no_grad():
            gamma_corrected = torch.sigmoid(torch.FloatTensor(gamma_logit))
        alpha_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(alpha_logit)))
        gamma_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(gamma_logit)))

        optimizer = Optimizer(verbose=False)
        optimizer.add([alpha_logit], 1e-2)
        optimizer.add([gamma_logit], 1e-3)
        optimizer.add(list(bias_correction.parameters()), 1e-3)

        n_steps_without_improvement = 0
        best_loss = np.inf
        for _ in tqdm.tqdm(range(max_n_iter), desc='Correcting atlas'):

            optimizer.zero_grad()

            alpha = torch.softmax(alpha_logit, dim=1)
            gamma = torch.sigmoid(gamma_logit)

            # Bias correction on atlas
            gamma_corrected = bias_correction.forward(gamma_logit)
            gamma_corrected = torch.clamp(gamma_corrected, lb, ub)

            # Methylation ratios should stay close to the original atlas
            w = weights_atlas * (1. + n_tissues * torch.mean(alpha[:, :n_known_tissues], dim=0).unsqueeze(1))
            w = w / torch.clamp(torch.std(R_atlas, dim=0), 0.005).unsqueeze(0)
            g2 = torch.sum(w * ((gamma[:n_known_tissues, :] - R_atlas) ** 2))

            # Reconstruction error of patients' profiles.
            # cfDNA profiles are reconstructed based on the corrected atlas
            R_reconstructed = torch.mm(alpha, gamma_corrected)
            g1 = torch.sum(weights_cfdna * ((R_reconstructed - R_cfdna) ** 2))

            # Regularisation on the bias terms
            u_l2 = torch.mean(bias_correction.tissue_bias ** 2)
            v_l2 = torch.mean(bias_correction.marker_bias ** 2)

            # Overall loss function
            loss = g1 + self.lambda1 * g2 + self.lambda2 * (u_l2 + v_l2)

            # Gradient backpropagation
            loss.backward()

            info['loss'].append(loss.item())

            # Update parameters
            #optimizer.step()
            optimizer.step(loss.item())

            # scheduler.step()
            if loss.item() >= best_loss:
                n_steps_without_improvement += 1
                if n_steps_without_improvement >= patience:
                    break
            else:
                best_loss = loss.item()
                n_steps_without_improvement = 0

        print(f'Final loss: {info["loss"][-1]}')

        R_corrected = gamma_corrected.cpu().data.numpy()
        diff1 = np.abs((torch.sigmoid(gamma_logit) - R_atlas).cpu().data.numpy())
        diff2 = np.abs(R_corrected[:n_known_tissues, :] - R_atlas.cpu().data.numpy())
        for j in range(diff1.shape[0]):
            d1 = np.mean(diff1[j, :]) * 100.
            d2 = np.mean(diff2[j, :]) * 100.
            print(f'[MD] Avg. meth. diff. for atlas entity {j}: {d1} % -> {d2} %')
        print(f'[MD] Avg. atlas std: {np.mean(np.std(R_atlas.cpu().data.numpy(), axis=0))} -> '
              f'{np.mean(np.std(R_corrected, axis=0))}')

        self.R_atlas = np.clip(R_corrected, 0, 1)
        self.D_atlas = D_atlas
        if len(self.R_atlas) > len(self.D_atlas):
            assert len(self.R_atlas) - len(self.D_atlas) == n_unknown_tissues
            D_atlas_ = [self.D_atlas]
            avg_depths = np.maximum(np.round(np.mean(D_atlas, axis=0)).astype(int), 1)
            for _ in range(n_unknown_tissues):
                D_atlas_.append(avg_depths[np.newaxis, :])
            self.D_atlas = np.concatenate(D_atlas_, axis=0)
            print(self.D_atlas.shape, self.D_atlas.dtype)

        return self

    def deconvolute(
            self,
            M_cfdna: np.ndarray,
            D_cfdna: np.ndarray
    ) -> np.ndarray:

        # Add pseudo-counts
        M_cfdna, D_cfdna = MetDecode.add_pseudo_counts(M_cfdna, D_cfdna)

        # Compute importance weights based on coverage
        weights_cfdna = self.compute_weights(D_cfdna)

        R_cfdna = M_cfdna / D_cfdna
        alpha = MetDecode.nnls(self.R_atlas, R_cfdna, weights_cfdna, 0)

        return alpha

    def set_atlas(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray
    ):
        # Add pseudo-counts
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)

        self.R_atlas = M_atlas / D_atlas
        self.D_atlas = D_atlas
