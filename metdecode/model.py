# -*- coding: utf-8 -*-
#
#  model.py
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
from scipy.optimize import nnls
from scipy.special import logit

from metdecode.deconvolution.lts import least_trimmed_squares
from metdecode.losses import Loss
from metdecode.optimizer import Optimizer
from metdecode.postprocessing.assignment import cell_type_assignment
from metdecode.postprocessing.wasserstein import ot_mapping


class Model:

    def __init__(
            self,
            p: float = 0.5,
            lambda1: float = 0.5,
            lambda2: float = 0.01,
            lambda3: float = 0.0,
            lts_ratio: float = 1.0,
            cta_ratio: float = 0.0,
            cta_type: str = 'perm',
            obs_criterion: str = 'l2',
            obs_criterion_eps: float = 0.02,  # 0.02
            obs_criterion_nu: float = 0.03067969,
            ref_criterion: str = 'l2',
            ref_criterion_eps: float = 0.02,  # 0.02
            ref_criterion_nu: float = 0.8394,
    ):
        self.p: float = p
        self.lambda1: float = lambda1
        self.lambda2: float = lambda2
        self.lambda3: float = lambda3
        self.lts_ratio: float = lts_ratio
        self.cta_ratio: float = cta_ratio
        self.cta_type: str = cta_type
        assert self.cta_type in {'perm', 'ot'}

        self.obs_criterion: Loss = Loss(
            criterion=obs_criterion,
            nu=obs_criterion_nu,
            eps=obs_criterion_eps
        )
        self.ref_criterion: Loss = Loss(
            criterion=ref_criterion,
            nu=ref_criterion_nu,
            eps=ref_criterion_eps
        )

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

    @staticmethod
    def init_alpha_and_gamma(
            R_atlas: np.ndarray,
            R_cfdna: np.ndarray,
            W_cfdna: np.ndarray,
            n_unknowns: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha_hat = Model.nnls(R_atlas, R_cfdna, W_cfdna)
        alpha_hat += 1e-6
        alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

        lb = np.min(R_atlas, axis=0)
        ub = np.max(R_atlas, axis=0)

        while n_unknowns > 0:
            residuals = np.dot(alpha_hat, R_atlas) - R_cfdna
            residuals = np.median(residuals, axis=0)
            r = (residuals <= 0).astype(float)
            r = lb + r * (ub - lb)
            R_atlas = np.concatenate((R_atlas, r[np.newaxis, :]), axis=0)

            alpha_hat = Model.nnls(R_atlas, R_cfdna, W_cfdna)
            alpha_hat += 1e-6
            alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

            n_unknowns -= 1

        alpha_logit = np.log(alpha_hat)
        gamma_logit = logit(R_atlas)

        return alpha_logit, gamma_logit

    def compute_weights(self, depths: np.ndarray) -> np.ndarray:
        depths = depths ** self.p
        return depths / np.sum(depths)

    def fit(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray,
            M_cfdna: np.ndarray,
            D_cfdna: np.ndarray,
            n_unknown_tissues: int = 0,
            infer: bool = True
    ) -> np.ndarray:
        return self.fit_(
            M_atlas, D_atlas, M_cfdna, D_cfdna, n_unknown_tissues=n_unknown_tissues,
            infer=infer
        )

    def deconvolute(
            self,
            M_cfdna: np.ndarray,
            D_cfdna: np.ndarray
    ) -> np.ndarray:

        M_atlas = self.R_atlas * self.D_atlas
        D_atlas = self.D_atlas
        return self.fit_(
            M_atlas, D_atlas, M_cfdna, D_cfdna, n_unknown_tissues=0,
            infer=False
        )

    def fit_(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray,
            M_cfdna: np.ndarray,
            D_cfdna: np.ndarray,
            infer: bool = True,
            n_unknown_tissues: int = 0,
            max_n_iter: int = 3000,
            patience: int = 100
    ) -> np.ndarray:

        # Add pseudo-counts
        M_atlas, D_atlas = Model.add_pseudo_counts(M_atlas, D_atlas)
        M_cfdna, D_cfdna = Model.add_pseudo_counts(M_cfdna, D_cfdna)

        # Compute methylation ratios
        R_atlas = torch.FloatTensor(M_atlas / D_atlas)
        R_cfdna = torch.FloatTensor(M_cfdna / D_cfdna)

        n_known_tissues = D_atlas.shape[0]
        n_markers = D_atlas.shape[1]
        n_tissues = n_known_tissues + n_unknown_tissues

        # Store atlas depths, extend matrix to account for unknown entities
        self.D_atlas = D_atlas
        if n_unknown_tissues > 0:
            D_atlas_ = [self.D_atlas]
            avg_depths = np.maximum(np.round(np.mean(D_atlas, axis=0)).astype(int), 1)
            for _ in range(n_unknown_tissues):
                D_atlas_.append(avg_depths[np.newaxis, :])
            self.D_atlas = np.concatenate(D_atlas_, axis=0)

        # Compute importance weights based on coverage
        weights_atlas = torch.FloatTensor(self.compute_weights(D_atlas))
        weights_cfdna = torch.FloatTensor(self.compute_weights(D_cfdna))

        print(f'[MD] Unknown tissues    : {n_unknown_tissues}')
        print(f'[MD] p coefficient      : {self.p}')
        print(f'[MD] lambda1            : {self.lambda1}')
        print(f'[MD] lambda2            : {self.lambda2}')

        info = {'loss': []}

        alpha_logit, gamma_logit = Model.init_alpha_and_gamma(
            R_atlas.cpu().data.numpy(), R_cfdna.cpu().data.numpy(),
            weights_cfdna.cpu().data.numpy(), n_unknown_tissues
        )
        alpha_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(alpha_logit)))
        gamma_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(gamma_logit)))

        # Initialize optimizers
        alpha_optimizer = Optimizer(verbose=False)
        alpha_optimizer.add([alpha_logit], 1e-2)
        gamma_optimizer = Optimizer(verbose=False)
        if infer:
            gamma_optimizer.add([gamma_logit], 1e-3)

        best_alpha = torch.softmax(alpha_logit, dim=1)
        best_gamma = torch.sigmoid(gamma_logit)

        running = True

        n_steps_without_improvement = 0
        best_loss = np.inf
        for it in tqdm.tqdm(range(max_n_iter), desc='Correcting atlas'):

            if not running:
                break

            for update_alpha in ([False, True] if infer else [True]):

                optimizer = alpha_optimizer if update_alpha else gamma_optimizer

                optimizer.zero_grad()

                alpha = torch.softmax(alpha_logit, dim=1)
                gamma = torch.sigmoid(gamma_logit)

                # Reconstruction error of patients' profiles.
                # cfDNA profiles are reconstructed based on the corrected atlas
                R_reconstructed = torch.mm(alpha, gamma)
                residuals = R_reconstructed - R_cfdna
                if not update_alpha:
                    loss = torch.sum(weights_cfdna * self.obs_criterion(residuals))
                else:
                    # LTS can remove marker regions that carry signals from unknown tissues
                    # not present in the reference atlas
                    if self.lts_ratio < 1.0:
                        loss = least_trimmed_squares(
                            self.obs_criterion, alpha, gamma, R_cfdna, weights_cfdna,
                            n_features=int(np.clip(round(self.lts_ratio * n_markers), 0, n_markers))
                        )
                    else:
                        w = weights_cfdna / torch.sum(weights_cfdna, dim=1).unsqueeze(1)
                        loss = torch.sum(torch.max(w * self.obs_criterion(residuals), dim=0).values)

                if not update_alpha:

                    # Methylation ratios should stay close to the original atlas
                    residuals = gamma[:n_known_tissues, :] - R_atlas
                    w = weights_atlas / torch.sum(weights_atlas, dim=1).unsqueeze(1)
                    loss = loss + self.lambda1 * torch.sum(torch.max(w * self.ref_criterion(residuals), dim=0).values)

                    # Methylation ratios should be close to 0 or 1
                    loss = loss + self.lambda2 * torch.mean(gamma * (1. - gamma))

                # Gradient backpropagation
                loss.backward()

                info['loss'].append(loss.item())

                # Update parameters
                optimizer.step(loss.item())

                # scheduler.step()
                if update_alpha:
                    # print(loss.item(), best_loss)
                    if loss.item() >= best_loss:
                        n_steps_without_improvement += 1
                        if n_steps_without_improvement >= patience:
                            running = False
                            break
                    else:
                        best_alpha = alpha
                        best_gamma = gamma
                        best_loss = loss.item()
                        n_steps_without_improvement = 0

        print(f'Final loss: {info["loss"][-1]}')
        R = best_gamma

        if self.cta_type == 'ot':
            plan = ot_mapping(R, R_atlas)
            R = self.cta_ratio * torch.mm(plan, R_atlas) + (1. - self.cta_ratio) * R
        else:
            R = self.cta_ratio * cell_type_assignment(R_atlas, R) + (1. - self.cta_ratio) * R

        # TODO: Optimal transport

        alpha = best_alpha.cpu().data.numpy()

        self.R_atlas = np.clip(R.cpu().data.numpy(), 0, 1)

        return alpha

    def set_atlas(
            self,
            M_atlas: np.ndarray,
            D_atlas: np.ndarray
    ):
        assert np.all(D_atlas > 0)
        self.R_atlas = M_atlas / D_atlas
        self.D_atlas = D_atlas
