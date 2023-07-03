# -*- coding: utf-8 -*-
# metdecode2.py
# author: Antoine Passemiers

import numpy as np
from scipy.optimize import nnls


class MetDecodeV2:

    def __init__(self,
                 R_methylated, R_depths, X_methylated, X_depths,
                 n_unknown_tissues: int = 0,
                 beta: float = 1.0
                 ):

        self.X_methylated = X_methylated
        self.X_depths = X_depths
        self.R_methylated = R_methylated
        self.R_depths = R_depths
        self.n_known_tissues = self.R_methylated.shape[0]
        self.n_unknown_tissues = n_unknown_tissues
        self.n_tissues = self.n_known_tissues + self.n_unknown_tissues
        self.beta: float = beta

        # Add pseudo-counts
        self.R_methylated, self.R_depths = MetDecodeV2.add_pseudo_counts(self.R_methylated, self.R_depths)
        self.X_methylated, self.X_depths = MetDecodeV2.add_pseudo_counts(self.X_methylated, self.X_depths)

        R = np.asarray([self.R_methylated, self.R_depths - self.R_methylated])
        self.R = np.transpose(R, (1, 2, 0)).reshape(self.R_depths.shape[0], 2 * self.R_depths.shape[1])
        X = np.asarray([self.X_methylated, self.X_depths - self.X_methylated])
        self.X = np.transpose(X, (1, 2, 0)).reshape(self.X_depths.shape[0], 2 * self.X_depths.shape[1])

    @staticmethod
    def add_pseudo_counts(methylated, depths, pc=0.8):
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
        meth_prior = np.outer(np.ones(len(row_meth)), col_meth)  # TODO: use survival function of Beta distribution instead
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

    def deconvolute(self) -> np.ndarray:

        R_atlas = self.R_methylated / self.R_depths
        R_cfdna = self.X_methylated / self.X_depths
        W_cfdna = self.X_depths / np.sum(self.X_depths)
        alpha_hat = MetDecodeV2.nnls(R_atlas, R_cfdna, W_cfdna)
        alpha_hat += 1e-6
        alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

        lb = np.min(R_atlas, axis=0)
        ub = np.max(R_atlas, axis=0)

        n_unknowns = self.n_unknown_tissues
        while n_unknowns > 0:
            residuals = np.dot(alpha_hat, R_atlas) - R_cfdna
            residuals = np.median(residuals, axis=0)
            r = (residuals <= 0).astype(float)
            r = lb + r * (ub - lb)
            R_atlas = np.concatenate((R_atlas, r[np.newaxis, :]), axis=0)

            alpha_hat = MetDecodeV2.nnls(R_atlas, R_cfdna, W_cfdna)
            alpha_hat += 1e-6
            alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

            n_unknowns -= 1

        return alpha_hat
