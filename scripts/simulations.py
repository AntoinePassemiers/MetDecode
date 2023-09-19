# -*- coding: utf-8 -*-
#
#  simulations.py
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

import enum
import os
import uuid

import numpy as np
import scipy.stats

from metdecode.evaluation import Evaluation
from metdecode.io import load_input_file
from metdecode.model import MetDecode

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas.tsv')
OUT_FOLDER = os.path.join(ROOT, 'sim-results')


class ExperimentType(enum.Enum):

    UNKNOWNS = enum.auto()
    COVERAGE = enum.auto()


def generate_mix_dataset(
        n_profiles: int = 48,
        n_unknown_tissues: int = 0,
        n_markers: int = 4441
) -> dict:

    n_known_tissues = 13
    n_tissues = n_known_tissues + n_unknown_tissues

    M_atlas, D_atlas, cell_type_names, row_names = load_input_file(ATLAS_FILEPATH)
    row_names = np.asarray(row_names, dtype=object)
    print(cell_type_names)
    assert len(cell_type_names) == len(M_atlas)
    assert len(cell_type_names) == len(D_atlas)
    assert M_atlas.shape[0] == n_known_tissues

    idx = np.arange(D_atlas.shape[1])
    np.random.shuffle(idx)
    idx = idx[:n_markers]
    M_atlas = M_atlas[:, idx]
    D_atlas = D_atlas[:, idx]
    row_names = row_names[idx]

    #assert R_depths.shape[0] == n_tissues
    #n_tissues = R_depths.shape[0]
    assert n_known_tissues <= n_tissues
    assert n_markers <= D_atlas.shape[1]

    R_depths = D_atlas + 1
    x_depths = np.mean(D_atlas, axis=0)
    X_depths = np.clip(scipy.stats.poisson.rvs(x_depths, size=(n_profiles, len(x_depths))), 1, np.inf)
    R_methylated_orig = M_atlas + 1

    gamma = R_methylated_orig / R_depths

    for i in range(n_unknown_tissues):
        unk = (np.random.rand(1, gamma.shape[1]) < 0.7).astype(float)
        gamma = np.concatenate((gamma, unk), axis=0)
        R_depths = np.concatenate((R_depths, np.median(R_depths, axis=0)[np.newaxis, :]), axis=0)

    alpha = np.zeros((n_profiles, n_tissues))
    prior = [0.7393, 3.0938, 8.2576, 3.8222, 1.7946, 4.6937, 2.6914, 29.6514]
    for i in range(n_unknown_tissues):
        prior.append(15)
    for i in range(len(alpha)):
        alpha_ = np.random.dirichlet(np.asarray(prior))
        j = np.random.randint(0, 6)
        alpha[i, j] = alpha_[0]
        alpha[i, 6:] = alpha_[1:]
    print(alpha)
    cancer_mask = np.zeros(n_tissues, dtype=bool)
    cancer_mask[:6] = True

    coverage_factor = 1
    R_methylated = np.random.binomial(np.round(R_depths * coverage_factor).astype(int), gamma)
    X_methylated = np.random.binomial(np.round(X_depths * coverage_factor).astype(int), np.dot(alpha, gamma), size=(n_profiles, R_depths.shape[1]))
    R_methylated = np.round(R_methylated / coverage_factor).astype(int)
    X_methylated = np.round(X_methylated / coverage_factor).astype(int)
    R_methylated = np.clip(R_methylated, 0, R_depths)
    X_methylated = np.clip(X_methylated, 0, X_depths)

    assert R_methylated.shape[1] == n_markers
    assert R_depths.shape[1] == n_markers
    assert alpha.shape == (n_profiles, n_tissues)
    assert gamma.shape == (n_tissues, n_markers)
    assert X_methylated.shape == (n_profiles, n_markers)
    assert X_depths.shape == (n_profiles, n_markers)
    assert R_methylated.shape == (n_tissues, n_markers)
    assert R_depths.shape == (n_tissues, n_markers)

    return {
        'X-methylated': X_methylated,
        'X-depths': X_depths,
        'R-methylated': R_methylated[:n_known_tissues],
        'R-depths': R_depths[:n_known_tissues],
        'Alpha': alpha,
        'Gamma': gamma[:n_known_tissues],
        'n-known-tissues': n_known_tissues,
        'n-unknown-tissues': n_tissues - n_known_tissues,
        'cancer-mask': cancer_mask,
        'row-names': row_names
    }


def evaluate(exp: ExperimentType, dataset: dict, evaluation: Evaluation, exp_id: str):
    n_known_tissues = dataset['R-methylated'].shape[0]

    args = [
        dataset['R-methylated'],
        dataset['R-depths'],
        dataset['X-methylated'],
        dataset['X-depths'],
    ]

    if exp == ExperimentType.UNKNOWNS:
        METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nc']
    else:
        METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nu']
    for method_name in METHOD_NAMES:
        if method_name == 'metdecode2-nc-nu':
            model = MetDecode(*args, n_unknown_tissues=0, beta=0)
        elif method_name == 'metdecode2-nu':
            model = MetDecode(*args, n_unknown_tissues=0, beta=1)
        elif method_name == 'metdecode2-nc':
            model = MetDecode(*args, n_unknown_tissues=1, beta=0)
        else:
            model = MetDecode(*args, n_unknown_tissues=1, beta=1)
        alpha_hat = model.deconvolute()
        evaluation.add(alpha_hat, dataset['Alpha'], dataset['Gamma'],
            dataset['X-methylated'], dataset['X-depths'], method_name, exp_id, n_known_tissues)


def run_simulation(exp: ExperimentType):

    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    dataset = generate_mix_dataset(
        n_unknown_tissues=1 if (exp == ExperimentType.UNKNOWNS) else 0,
        n_profiles=100,
        n_markers=1000
    )
    evaluation = Evaluation()
    evaluate(exp, dataset, evaluation, f'sim')
    out_folder = os.path.join(OUT_FOLDER, 'unknowns' if (exp == ExperimentType.UNKNOWNS) else 'coverage')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    evaluation.save(os.path.join(out_folder, f'results-{uuid.uuid4()}.pkl'))


if __name__ == '__main__':

    # Run simulation experiments to test the modelling of unknown tissues
    for k in range(5):
        print(f'Running {ExperimentType.UNKNOWNS} ({k + 1} / 30)')
        run_simulation(ExperimentType.UNKNOWNS)

    # Run simulation experiments to test the modelling of unknown tissues
    for k in range(5):
        print(f'Running {ExperimentType.COVERAGE} ({k + 1} / 30)')
        run_simulation(ExperimentType.COVERAGE)
