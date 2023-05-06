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
from sklearn.decomposition import PCA

from metdecode.optimizer import Optimizer


class BiasCorrection(torch.nn.Module):

    def __init__(
            self,
            n_tissues: int,
            n_markers: int,
            multiplicative: bool,
    ):
        torch.nn.Module.__init__(self)
        self.n_tissues: int = n_tissues
        self.multiplicative: bool = multiplicative

        self.tissue_bias = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=n_tissues)))
        self.marker_bias = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, 0.01, size=n_markers)))

    def forward(self, R_logit: torch.Tensor) -> torch.Tensor:

        # Atlas correction
        if self.multiplicative:
            bias_logit = self.marker_bias.unsqueeze(0) * self.tissue_bias.unsqueeze(1)
            R_corrected = torch.sigmoid(R_logit + bias_logit)
        else:
            R_corrected = torch.sigmoid(R_logit) ** (1. + self.tissue_bias.unsqueeze(1))

        return R_corrected

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if m.weight.size()[1] == 1:
                torch.nn.init.xavier_uniform(m.weight)
            else:
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            m.bias.data.fill_(0.0001)


class MetDecode:

    def __init__(
            self,
            max_correction: float = 0.1,
            p: float = 1.0,
            lambda1: float = 0.001,
            lambda2: float = 0.001,
            coverage_rcw: bool = False,
            multiplicative: bool = False,
            z1: float = 1.0,
            z2: float = 0.1,
            alpha_weighting: bool = True,
            std_weighting: bool = True,
            unk_clip: bool = True,
            unk_qt: bool = True,
            unk_bound: bool = False,
            unk_q: float = 0.05,
            init_unk_prop: float = 0.1,
            verbose: bool = True
    ):
        self.max_correction: float = max_correction
        self.p: float = p
        self.lambda1: float = lambda1
        self.lambda2: float = lambda2
        self.coverage_rcw: bool = coverage_rcw
        self.multiplicative: bool = multiplicative
        self.z1: float = z1
        self.z2: float = z2
        self.alpha_weighting: bool = alpha_weighting
        self.std_weighting: bool = std_weighting
        self.unk_clip: bool = unk_clip
        self.unk_qt: bool = unk_qt
        self.unk_bound: bool = unk_bound
        self.unk_q: float = unk_q
        self.init_unk_prop: float = init_unk_prop
        self.verbose: bool = verbose

        self.R_atlas: Optional[np.ndarray] = None
        self.D_atlas: Optional[np.ndarray] = None

    @staticmethod
    def add_pseudo_counts(methylated, depths, pc=0.8):
        original_methylated = methylated
        original_depths = depths
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
        assert np.all(np.abs(original_methylated - methylated) <= 1)
        assert np.all(np.abs(original_depths - depths) <= 1)

        return methylated, depths

    @staticmethod
    def quantile_transform(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        inv_idx1 = np.argsort(np.argsort(x1))
        idx2 = np.argsort(x2)
        return x2[idx2[inv_idx1]]

    @staticmethod
    def rcw(mat: np.ndarray) -> np.ndarray:
        return np.log(2 + mat)

    @staticmethod
    def nnls(R_atlas: np.ndarray, R_cfdna: np.ndarray, W_cfdna: np.ndarray) -> np.ndarray:
        n_samples = len(R_cfdna)
        alpha_hat = []
        for i in range(n_samples):
            A = (np.sqrt(W_cfdna[np.newaxis, i, :]) * R_atlas).T
            b = np.sqrt(W_cfdna[i, :]) * R_cfdna[i, :]
            x, residuals = nnls(A, b, maxiter=3000)
            alpha_hat.append(x)
        alpha_hat = np.asarray(alpha_hat)
        alpha_hat /= alpha_hat.sum(axis=1)[:, np.newaxis]
        return alpha_hat

    def init_alpha_and_gamma(
            self,
            R_atlas: np.ndarray,
            R_cfdna: np.ndarray,
            W_cfdna: np.ndarray,
            n_unknowns: int,
            init_unk_prop: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_known = R_atlas.shape[0]

        # Bounds on the unknown tissues
        lb = np.min(R_atlas, axis=0) - 1e-4
        ub = np.max(R_atlas, axis=0) + 1e-4

        # Adding unknown tissues to the atlas
        if n_unknowns > 0:
            for _ in range(n_unknowns):
                alpha_hat = MetDecode.nnls(R_atlas, R_cfdna, W_cfdna)
                residuals = np.mean(np.dot(alpha_hat, R_atlas) - R_cfdna, axis=0)

                unk = 1. - residuals
                if self.unk_bound:
                    unk = (unk >= 0).astype(float) * (1.0 - 2 * self.unk_q) + self.unk_q
                    unk = lb + unk * (ub - lb)
                if self.unk_qt:
                    unk = MetDecode.quantile_transform(unk, np.median(R_atlas, axis=0))
                if self.unk_clip:
                    unk = np.clip(unk, lb, ub)
                else:
                    unk = np.clip(unk, 0, 1)

                R_atlas = np.concatenate((R_atlas, unk[np.newaxis, :]), axis=0)

        # Deconvolve using full atlas
        alpha_hat = MetDecode.nnls(R_atlas, R_cfdna, W_cfdna)
        alpha_hat += 1e-7
        alpha_hat[:, :n_known] /= np.sum(alpha_hat[:, :n_known], axis=1)[:, np.newaxis]
        alpha_hat[:, :n_known] *= (1. - init_unk_prop)
        alpha_hat[:, n_known:] /= np.sum(alpha_hat[:, n_known:], axis=1)[:, np.newaxis]
        alpha_hat[:, n_known:] *= init_unk_prop

        # Convert to log-scale
        alpha_hat += 1e-6
        alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]
        alpha_logit = np.log(alpha_hat)
        gamma_logit = logit(R_atlas)
        return alpha_logit, gamma_logit

    def compute_weights(self, depths: np.ndarray) -> np.ndarray:
        depths = np.clip(depths, 0, 200)
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

        n_known_tissues = D_atlas.shape[0]
        n_tissues = n_known_tissues + n_unknown_tissues

        # Add pseudo-counts
        M_atlas, D_atlas = MetDecode.add_pseudo_counts(M_atlas, D_atlas)
        M_cfdna, D_cfdna = MetDecode.add_pseudo_counts(M_cfdna, D_cfdna)

        # Compute methylation ratios
        R_atlas = torch.FloatTensor(M_atlas / D_atlas)
        R_cfdna = torch.FloatTensor(M_cfdna / D_cfdna)

        pca = PCA()
        coords = pca.fit_transform(R_atlas)
        plt.scatter(coords[:6, 0], coords[:6, 1], color='blue', marker='x')
        plt.scatter(coords[6:, 0], coords[6:, 1], color='blue', marker='o')
        coords = pca.transform(R_cfdna)
        plt.scatter(coords[:, 0], coords[:, 1], color='orange')
        plt.show()

        # Store atlas depths, extend matrix to account for unknown entities
        self.D_atlas = D_atlas
        if n_unknown_tissues > 0:
            D_atlas_ = [self.D_atlas]
            avg_depths = np.maximum(np.round(np.mean(D_atlas, axis=0)).astype(int), 1)
            for _ in range(n_unknown_tissues):
                D_atlas_.append(avg_depths[np.newaxis, :])
            self.D_atlas = np.concatenate(D_atlas_, axis=0)

        # Compute importance weights based on coverage
        weights_atlas = torch.FloatTensor(self.compute_weights(self.D_atlas))
        weights_cfdna = torch.FloatTensor(self.compute_weights(D_cfdna))

        alpha_logit, gamma_logit = self.init_alpha_and_gamma(
            R_atlas.cpu().data.numpy(), R_cfdna.cpu().data.numpy(),
            weights_cfdna.cpu().data.numpy(), n_unknown_tissues,
            init_unk_prop=self.init_unk_prop,
        )
        with torch.no_grad():
            R_atlas = torch.sigmoid(torch.FloatTensor(gamma_logit))

        # Compute lower and upper bounds on methylation ratios
        mc = self.max_correction
        with torch.no_grad():
            lb = torch.clamp(R_atlas - mc, 0, 1)  # Lower bounds
            ub = torch.clamp(R_atlas + mc, 0, 1)  # Upper bounds

        bias_correction = BiasCorrection(n_tissues, D_cfdna.shape[1], self.multiplicative)

        if self.verbose:
            print(f'[MD] Unknown tissues    : {n_unknown_tissues}')
            print(f'[MD] Maximum correction : {self.max_correction}')
            print(f'[MD] p coefficient      : {self.p}')

        info = {'loss': []}

        with torch.no_grad():
            gamma_corrected = torch.sigmoid(torch.FloatTensor(gamma_logit))
        gamma_best = torch.clone(gamma_corrected)
        alpha_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(alpha_logit)))
        gamma_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(gamma_logit)))

        optimizer = Optimizer(verbose=False)
        optimizer.add([alpha_logit], 1e-2)
        optimizer.add([gamma_logit], 1e-3)
        optimizer.add(list(bias_correction.parameters()), 1e-3)

        n_steps_without_improvement = 0
        best_loss = np.inf
        for it in tqdm.tqdm(range(max_n_iter), desc='Correcting atlas'):

            optimizer.zero_grad()

            alpha = torch.softmax(alpha_logit, dim=1)
            gamma = torch.sigmoid(gamma_logit)

            # Bias correction on atlas
            gamma_corrected = bias_correction.forward(gamma_logit)
            gamma_corrected = torch.clamp(gamma_corrected, lb, ub)

            # Methylation ratios should stay close to the original atlas
            w = weights_atlas
            if self.alpha_weighting:
                w = w * (self.z1 + n_tissues * torch.mean(alpha, dim=0).unsqueeze(1))
            if self.std_weighting:
                w = w / torch.clamp(torch.std(R_atlas, dim=0), self.z2).unsqueeze(0)
            w = w / torch.sum(w)
            g2 = torch.max(w * ((gamma - R_atlas) ** 2))

            # Reconstruction error of patients' profiles.
            # cfDNA profiles are reconstructed based on the corrected atlas
            R_reconstructed = torch.mm(alpha, gamma_corrected)
            g1 = torch.max(weights_cfdna * ((R_reconstructed - R_cfdna) ** 2))

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
            if it >= 100:
                if loss.item() >= best_loss:
                    n_steps_without_improvement += 1
                    if n_steps_without_improvement >= patience:
                        break
                else:
                    gamma_best = gamma_corrected
                    best_loss = loss.item()
                    n_steps_without_improvement = 0

        if self.verbose:
            print(f'Final loss: {info["loss"][-1]}')

        R_corrected = gamma_best.cpu().data.numpy()
        diff1 = np.abs((torch.sigmoid(gamma_logit) - R_atlas).cpu().data.numpy())
        diff2 = np.abs(R_corrected - R_atlas.cpu().data.numpy())
        if self.verbose:
            for j in range(diff1.shape[0]):
                d1 = np.mean(diff1[j, :]) * 100.
                d2 = np.mean(diff2[j, :]) * 100.
                # print(f'[MD] Avg. meth. diff. for atlas entity {j}: {d1} % -> {d2} %')
                print(f'[MD] Avg. meth. diff. for atlas entity {j}: {d2} %')
            print(f'[MD] Avg. atlas std: {np.mean(np.std(R_atlas.cpu().data.numpy(), axis=0))} -> '
                  f'{np.mean(np.std(R_corrected, axis=0))}')

        # print('TEST', np.median(np.nan_to_num((R_corrected - lb.cpu().data.numpy()) / (ub - lb).cpu().data.numpy(), nan=0.5)))

        self.R_atlas = np.clip(R_corrected, 0, 1)

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
        alpha = MetDecode.nnls(self.R_atlas, R_cfdna, weights_cfdna)

        return alpha

    def set_atlas(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray
    ):
        assert np.all(D_atlas > 0)
        self.R_atlas = M_atlas / D_atlas
        self.D_atlas = D_atlas
