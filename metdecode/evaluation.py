# -*- coding: utf-8 -*-
#
#  evaluation.py
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

import pickle
from typing import Any, Dict

import numpy as np
import scipy.stats


class Evaluation:
    """Data structure for storing results of simulation experiments.

    Attributes:
        data: Dict, where each key is an algorithm name and each value a sub-dict.
            For each sub-dict, each key is an experiment name and each value a sub-sub-dict.
            For each sub-sub-dict, each key is a string and each value an experiment-specific
            piece of information, such as the reference atlas used, the estimated cell type contributions
            or the Pearson correlation coefficients used to score the algorithm.
        exp_ids: Set containing all the experiment names.
    """

    def __init__(self):
        self.data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.exp_ids: set = set()

    def get(self, method_name: str, metric: str) -> Dict[Any, np.ndarray]:
        """Get a piece of information for a given algorithm across all experiments.

        Args:
            method_name: Name of the algorithm.
            metric: Name of the queried information.

        Returns:
            A dict, where each key is an experiment name, and each value the requested piece of information
                for that specific experiment.
        """
        metric = metric.replace('avg-', '')  # TODO
        values = {}
        for i, exp_id in enumerate(self.exp_ids):
            if exp_id in self.data[method_name]:
                values[exp_id] = self.data[method_name][exp_id][metric]
        return values

    def add(self, alpha_pred: np.ndarray, alpha: np.ndarray, gamma: np.ndarray,
            X_methylated: np.ndarray, X_depths: np.ndarray,
            method_name: str, exp_id: str, n_known_tissues: int):
        """Add results of a simulation experiment.

        Args:
            alpha_pred: Estimated cell type contributions, stored as a matrix of shape `(n_samples, n_tissues)`.
            alpha: Ground-truth cell type contributions, stored as a matrix of shape `(n_samples, n_tissues)`.
            gamma: Methylation ratios found in the reference atlas, stored as a matrix of shape `(n_tissues, n_markers)`.
            X_methylated: Methylation counts (number of methylated CpG sites spanned by reads within a specific marker
                region, for a specific sample), stored as a matrix of shape `(n_samples, n_markers)`.
            X_depths: Total counts (number of CpG sites, methylated or not, spanned by reads within a specific marker
                region, for a specific sample), stored as a matrix of shape `(n_samples, n_markers)`.
            method_name: Algorithm name.
            exp_id: Experiment name.
            n_known_tissues: Number of known tissues / cell types in the reference atlas.
        """

        assert len(gamma.shape) == 2
        self.exp_ids.add(exp_id)
        if method_name not in self.data:
            self.data[method_name] = {}
        alpha_pred_ = alpha_pred
        alpha_pred = alpha_pred[:, :n_known_tissues]

        # Score the algorithm against the ground-truth data
        mse = Evaluation.mse(alpha_pred, alpha)
        pearson = Evaluation.pearson(alpha_pred, alpha)
        spearman = Evaluation.spearman(alpha_pred, alpha)
        chi2 = Evaluation.chi_squared_distance(alpha_pred, alpha)
        ranking_score = Evaluation.ranking_score(alpha_pred, alpha)

        # Store the results as a dict
        self.data[method_name][exp_id] = {
            'alpha': alpha,
            'alpha-pred': alpha_pred_,
            'gamma': gamma,
            'x-depths': X_depths,
            'x-methylated': X_methylated,
            'mse': mse,
            'pearson': pearson,
            'spearman': spearman,
            'known-fraction-diff': Evaluation.known_fraction_diff(alpha_pred, alpha),
            'chi2-distance': chi2,
            'ranking_score': ranking_score
        }

    def save(self, filepath: str):
        """Save the results.

        Args:
            filepath: Pickle file where to store the results.
        """
        with open(filepath, 'wb') as f:
            pickle.dump((self.data, self.exp_ids), f)

    @staticmethod
    def load(filepath: str) -> 'Evaluation':
        """Load the results from a pickle file.

        Args:
            filepath: Pickle file where to load the results from.

        Returns:
            An `Evaluation` object.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        evaluation = Evaluation()
        evaluation.data = data[0]
        evaluation.exp_ids = data[1]
        return evaluation

    @staticmethod
    def mse(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute mean squared error between estimated and ground-truth cell type contributions.

         Returns:
             An array of size `(n_samples,)` containing the mean squared error per sample.
        """
        return np.mean((alpha[:, :alpha_pred.shape[1]] - alpha_pred) ** 2, axis=1)

    @staticmethod
    def pearson(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation coefficient between estimated and ground-truth cell type contributions.

         Returns:
             An array of size `(n_tissues,)` containing the correlation coefficient per reference cell type.
        """
        scores = []
        for j in range(alpha_pred.shape[1]):
            scores.append(scipy.stats.pearsonr(alpha_pred[:, j], alpha[:, j])[0])
        return np.nan_to_num(scores)

    @staticmethod
    def spearman(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute Spearman correlation coefficient between estimated and ground-truth cell type contributions.

         Returns:
             An array of size `(n_tissues,)` containing the correlation coefficient per reference cell type.
        """
        scores = []
        for j in range(alpha_pred.shape[1]):
            scores.append(scipy.stats.spearmanr(alpha_pred[:, j], alpha[:, j])[0])
        return np.nan_to_num(scores)

    @staticmethod
    def chi_squared_distance(alpha_pred: np.ndarray, alpha: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Compute chi² distance between estimated and ground-truth cell type contributions.

         Returns:
             An array of size `(n_tissues,)` containing the distance per reference cell type.
        """
        alpha_pred = alpha_pred + eps
        alpha = alpha[:, :alpha_pred.shape[1]] + eps
        return 0.5 * np.sum((alpha - alpha_pred) ** 2 / (alpha + alpha_pred), axis=1)

    @staticmethod
    def known_fraction_diff(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute the difference between the total contribution from known cell types
        (cell types actually present in the atlas) and the total estimated contributions.
        When the number of unknowns is 0, then the total contributions will necessarily be 1
        and therefore the difference will be 0. This metric is relevant only when the number
        of unknowns is > 0.

        Returns:
             An array of size `(n_samples,)` containing the distance per sample.
        """
        p1 = np.sum(alpha_pred, axis=1)
        p2 = np.sum(alpha[:, :alpha_pred.shape[1]], axis=1)
        return np.abs(p1 - p2)

    @staticmethod
    def ranking_score(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute the ranking scores.

        We define the ranking score as the probability that A > B both in the ground-truth and the estimations,
        or A < B both in the ground-truth and the estimations. In other terms, this score captures the probability
        that the algorithm correctly ranks two cell types based on their contributions to the mixture.

        Returns:
             An array of size `(n_samples,)` containing the score per sample.
        """
        alpha = alpha[:, :alpha_pred.shape[1]]
        R1 = np.greater_equal(alpha[:, :, np.newaxis], alpha[:, np.newaxis, :])
        R2 = np.greater_equal(alpha_pred[:, :, np.newaxis], alpha_pred[:, np.newaxis, :])
        return np.mean(np.equal(R1, R2), axis=(1, 2))
