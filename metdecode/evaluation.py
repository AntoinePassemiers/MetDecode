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
from typing import Iterable, List, Any, Dict, Generator

import numpy as np
import scipy.stats


class OrderedSet:

    def __init__(self, data: Iterable = tuple()):
        self._list: List[Any] = []
        self._dict: Dict[Any] = {}
        for element in data:
            self.add(element)

    def add(self, element: Any):
        if element not in self._dict:
            i = len(self._list)
            self._dict[element] = i
            self._list.append(element)

    def remove(self, element):
        i = self._dict.get(element, None)
        if i is not None:
            self._list.pop(i)
            del self._dict[element]

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, item: Any) -> Any:
        return self._list[item]

    def __setitem__(self, key: Any, value: Any):
        self._list[key] = value

    def __iter__(self) -> Generator[Any, None, None]:
        for element in self._list:
            yield element


class Evaluation:

    METRICS: List[str] = [
        'mse',
        'pearson',
        'spearman',
        'known-fraction-diff',
        'chi2-distance',
        'atlas-diff'
    ]

    def __init__(self):
        self.data: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.exp_ids: OrderedSet = OrderedSet()

    def get(self, method_name: str, metric: str) -> Dict[Any, np.ndarray]:
        metric = metric.replace('avg-', '')  # TODO
        values = {}
        for i, exp_id in enumerate(self.exp_ids):
            if exp_id in self.data[method_name]:
                values[exp_id] = self.data[method_name][exp_id][metric]
        return values

    def add(self, alpha_pred: np.ndarray, alpha: np.ndarray, gamma_pred: np.ndarray, gamma: np.ndarray,
            X_methylated: np.ndarray, X_depths: np.ndarray,
            method_name: str, exp_id: str, n_known_tissues: int):

        assert len(gamma_pred.shape) == 2
        assert len(gamma.shape) == 2

        self.exp_ids.add(exp_id)
        if method_name not in self.data:
            self.data[method_name] = {}
        alpha_pred_ = alpha_pred
        alpha_pred = alpha_pred[:, :n_known_tissues]
        mse = Evaluation.mse(alpha_pred, alpha)
        pearson = Evaluation.pearson(alpha_pred, alpha)
        spearman = Evaluation.spearman(alpha_pred, alpha)
        chi2 = Evaluation.chi_squared_distance(alpha_pred, alpha)
        ranking_score = Evaluation.ranking_score(alpha_pred, alpha)
        self.data[method_name][exp_id] = {
            'alpha': alpha,
            'alpha-pred': alpha_pred_,
            'gamma': gamma,
            'gamma-pred': gamma_pred,
            'x-depths': X_depths,
            'x-methylated': X_methylated,
            'mse': mse,
            'pearson': pearson,
            'spearman': spearman,
            'known-fraction-diff': Evaluation.known_fraction_diff(alpha_pred, alpha),
            'chi2-distance': chi2,
            'ranking_score': ranking_score,
            'atlas-diff': np.mean((gamma - gamma_pred[:gamma.shape[0], :]) ** 2)
        }

        #print(f'{method_name}: {np.mean(mse)} MSE, {np.mean(chi2)} chi2, '
        #      f'{ranking_score} ranking score, '
        #      f'{np.mean(pearson)} Pearson, {np.mean(spearman)} Spearman')

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump((self.data, self.exp_ids), f)

    """
    def save(self, filepath: str, sep: str = '\t'):
        method_names = list(self.data.keys())
        with open(filepath, 'w') as f:
            f.write(f'Run')
            for metric in Evaluation.METRICS:
                for method_name in method_names:
                    f.write(f'{sep}{method_name}_{metric}')
            f.write('\n')
            for exp_id in self.exp_ids:
                f.write(exp_id)
                for metric in Evaluation.METRICS:
                    for method_name in method_names:
                        value = np.nan
                        if exp_id in self.data[method_name]:
                            value = self.data[method_name][exp_id][metric]
                        f.write(f'{sep}{value}')
                f.write('\n')
    """

    def load(filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        evaluation = Evaluation()
        evaluation.data = data[0]
        evaluation.exp_ids = data[1]
        return evaluation

    """
    @staticmethod
    def load(filepath: str) -> 'Evaluation':
        evaluation = Evaluation()
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header = lines[0].rstrip().split('\t')[1:]
        for line in lines[1:]:
            elements = line.rstrip().split('\t')
            exp_id = elements[0]
            evaluation.exp_ids.add(exp_id)
            elements = elements[1:]
            for i in range(len(elements)):
                method_name, metric = header[i].split('_')
                if method_name not in evaluation.data:
                    evaluation.data[method_name] = {}
                if exp_id not in evaluation.data[method_name]:
                    evaluation.data[method_name][exp_id] = {}
                evaluation.data[method_name][exp_id][metric] = float(elements[i])
        return evaluation
    """

    @staticmethod
    def mse(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return np.mean((alpha[:, :alpha_pred.shape[1]] - alpha_pred) ** 2, axis=1)

    @staticmethod
    def pearson(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        m = alpha_pred.shape[0]
        scores = []
        for j in range(alpha_pred.shape[1]):
            scores.append(scipy.stats.pearsonr(alpha_pred[:, j], alpha[:, j])[0])
        return np.nan_to_num(scores)

    @staticmethod
    def spearman(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        m = alpha_pred.shape[0]
        scores = []
        for j in range(alpha_pred.shape[1]):
            scores.append(scipy.stats.spearmanr(alpha_pred[:, j], alpha[:, j])[0])
        return np.nan_to_num(scores)

    @staticmethod
    def chi_squared_distance(alpha_pred: np.ndarray, alpha: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        alpha_pred = alpha_pred + eps
        alpha = alpha[:, :alpha_pred.shape[1]] + eps
        return 0.5 * np.sum((alpha - alpha_pred) ** 2 / (alpha + alpha_pred), axis=1)

    @staticmethod
    def known_fraction_diff(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        p1 = np.sum(alpha_pred, axis=1)
        p2 = np.sum(alpha[:, :alpha_pred.shape[1]], axis=1)
        return np.asarray([p1, p2])

    @staticmethod
    def ranking_score(alpha_pred: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = alpha[:, :alpha_pred.shape[1]]
        R1 = np.greater_equal(alpha[:, :, np.newaxis], alpha[:, np.newaxis, :])
        R2 = np.greater_equal(alpha_pred[:, :, np.newaxis], alpha_pred[:, np.newaxis, :])
        return np.mean(np.equal(R1, R2), axis=(1, 2))