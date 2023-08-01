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

import os
import pickle
import uuid

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from metdecode.io import load_input_file
from scipy.special import expit

from metdecode.evaluation import Evaluation
from metdecode.md2 import MetDecodeV2
from metdecode.model import Model


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas_insil.tsv')


def generate_mix_dataset(
        n_profiles: int = 48,
        n_unknown_tissues: int = 0,
        n_markers: int = 4441
) -> dict:

    n_known_tissues = 13
    n_tissues = n_known_tissues + n_unknown_tissues

    M_cfdna, D_cfdna, sample_names, _ = load_input_file(os.path.join(DATA_FOLDER, 'bothbatch.tims.txt'))
    M_atlas, D_atlas, cell_type_names, _ = load_input_file(ATLAS_FILEPATH)
    print(cell_type_names)
    assert len(cell_type_names) == len(M_atlas)
    assert len(cell_type_names) == len(D_atlas)
    assert M_atlas.shape[0] == n_known_tissues

    idx = np.arange(D_cfdna.shape[1])
    np.random.shuffle(idx)
    idx = idx[:n_markers]
    M_cfdna = M_cfdna[:, idx]
    D_cfdna = D_cfdna[:, idx]
    M_atlas = M_atlas[:, idx]
    D_atlas = D_atlas[:, idx]

    #assert R_depths.shape[0] == n_tissues
    #n_tissues = R_depths.shape[0]
    assert n_known_tissues <= n_tissues
    assert n_markers <= D_atlas.shape[1]

    R_depths = D_atlas + 1
    x_depths = np.round(np.mean(D_cfdna + 1, axis=0)[np.newaxis, :]).astype(int)
    X_depths = np.repeat(x_depths, n_profiles, axis=0).astype(int)
    R_methylated_orig = M_atlas + 1

    gamma = R_methylated_orig / R_depths

    for i in range(n_unknown_tissues):
        unk = (np.random.rand(1, gamma.shape[1]) < 0.7).astype(float)
        gamma = np.concatenate((gamma, unk), axis=0)
        R_depths = np.concatenate((R_depths, np.median(R_depths, axis=0)[np.newaxis, :]), axis=0)

    #gamma = (gamma > np.median(gamma, axis=0)[np.newaxis, :]).astype(float)

    alpha = np.zeros((n_profiles, n_tissues))
    prior = [0.7393, 3.0938, 8.2576, 3.8222, 1.7946, 4.6937, 2.6914, 29.6514]
    for i in range(n_unknown_tissues):
        prior.append(15)
    for i in range(len(alpha)):
        alpha_ = np.random.dirichlet(np.asarray(prior))
        j = np.random.randint(0, 6)
        alpha[i, j] = alpha_[0]
        alpha[i, 6:] = alpha_[1:]
    cancer_mask = np.zeros(n_tissues, dtype=bool)
    cancer_mask[:6] = True

    gamma_distorted = gamma
    gamma_prime_clean = gamma

    coverage_factor = 1
    R_methylated = np.random.binomial(np.round(R_depths * coverage_factor).astype(int), gamma_distorted)
    X_methylated = np.random.binomial(np.round(X_depths * coverage_factor).astype(int), np.dot(alpha, gamma), size=(n_profiles, R_depths.shape[1]))
    R_methylated = np.round(R_methylated / coverage_factor).astype(int)
    X_methylated = np.round(X_methylated / coverage_factor).astype(int)
    R_methylated = np.clip(R_methylated, 0, R_depths)
    X_methylated = np.clip(X_methylated, 0, X_depths)

    assert R_methylated.shape[1] == n_markers
    assert R_depths.shape[1] == n_markers

    is_female = np.zeros(n_profiles, dtype=bool)
    age = np.full(n_profiles, 20)
    enzymatic = np.ones(n_profiles, dtype=bool)

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
        'Gamma-prime-clean': gamma_prime_clean[:n_known_tissues],
        'Gamma-prime-noisy': gamma_distorted[:n_known_tissues],
        'n-known-tissues': n_known_tissues,
        'age': age,
        'is-female': is_female,
        'enzymatic': enzymatic,
        'n-unknown-tissues': n_tissues - n_known_tissues,
        'cancer-mask': cancer_mask
    }


def evaluate(dataset: dict, evaluation: Evaluation, exp_id: str):
    n_profiles = dataset['X-depths'].shape[0]

    n_unknown_tissues = dataset['n-unknown-tissues']
    n_known_tissues = dataset['R-methylated'].shape[0]

    args = [
        dataset['R-methylated'],
        dataset['R-depths'],
        dataset['X-methylated'],
        dataset['X-depths'],
    ]

    # METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nu', 'metdecode2-nc', 'metdecode2']
    METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nc']

    for method_name in METHOD_NAMES:

        print(f'Running method "{method_name}"')

        gamma_hat = dataset['Gamma-prime-noisy']
        if method_name == 'metdecode2':
            model = MetDecodeV2(*args, n_unknown_tissues=1, beta=1)
            Alpha_hat = model.deconvolute()
        elif method_name == 'metdecode2-nu':
            model = MetDecodeV2(*args, n_unknown_tissues=0, beta=1)
            Alpha_hat = model.deconvolute()
        elif method_name == 'metdecode2-nc':
            model = MetDecodeV2(*args, n_unknown_tissues=1, beta=0)
            Alpha_hat = model.deconvolute()
        elif method_name == 'metdecode2-nc-nu':
            model = MetDecodeV2(*args, n_unknown_tissues=0, beta=0)
            Alpha_hat = model.deconvolute()
        else:
            raise NotImplementedError(f'Unknown method "{method_name}"')
        print(Alpha_hat.shape)

        evaluation.add(Alpha_hat, dataset['Alpha'], gamma_hat, dataset['Gamma'],
            dataset['X-methylated'], dataset['X-depths'], method_name, exp_id, n_known_tissues)


def main():

    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUT_FOLDER = os.path.join(ROOT, 'results')
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    N_REPEATS = 1

    dataset = generate_mix_dataset(
        n_unknown_tissues=1,
        n_profiles=1000,
        n_markers=1000
    )
    evaluation = Evaluation()
    evaluate(dataset, evaluation, f'sim')
    evaluation.save(f'results2-{uuid.uuid4()}.pkl')

    colors = [
        'slateblue', 'mediumvioletred', 'goldenrod', 'darkseagreen'
    ]

    #pretty_names = [r'$\beta=0, unk=0$', r'$\beta=1, unk=0$', r'$\beta=0, unk=1$', r'$\beta=1, unk=1$']
    pretty_names = [r'$\beta=0, unk=0$', r'$\beta=0, unk=1$']
    #methods = ['metdecode2-nc-nu', 'metdecode2-nu', 'metdecode2-nc', 'metdecode2']
    methods = ['metdecode2-nc-nu', 'metdecode2-nc']

    cell_type_names = [
        'BRCA', 'CEAD+CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T-cell', 'CD8+ T-cell',
        'Erythroblast', 'Monocyte', 'Natural killer cell', 'Neutrophil']
    fig = plt.figure(figsize=(12, 9))
    for j in range(11):
        plt.subplot(3, 4, j + 1)
        width = 0

        for m, method_name in enumerate(methods):
            max_correction = 0 if (m == 0) else 0.1
            alpha = evaluation.get(method_name, 'alpha')['sim']
            alpha_pred = evaluation.get(method_name, 'alpha-pred')['sim']
            if j == 0:
                xs, ys = alpha[:, j], alpha_pred[:, j]
            elif j == 1:
                xs, ys = np.sum(alpha[:, [1, 2]], axis=1), np.sum(alpha_pred[:, [1, 2]], axis=1)
            else:
                xs, ys = alpha[:, j+1], alpha_pred[:, j+1]
            xs, ys = 100 * xs, 100 * ys
            plt.scatter(xs, ys, color=colors[m], label=pretty_names[m], alpha=0.5, s=12)
            width = max(width, 1.1 * max(np.max(xs), np.max(ys)))

        plt.plot([0, width], [0, width], color='grey', linewidth=1, linestyle='--', alpha=0.5)
        plt.xlim(-0.05 * width, width)
        plt.ylim(-0.05 * width, width)
        plt.title(cell_type_names[j], fontname='Arial')
        if j == 0:
            plt.legend(prop={'size': 10})
    plt.tight_layout()
    plt.savefig('sim.png', dpi=300)

    print('')
    for i, method_name in enumerate(methods):
        mse = np.mean(evaluation.data[method_name][f'sim']['mse'])
        chi2 = np.mean(evaluation.data[method_name][f'sim']['chi2-distance'])
        pearson = np.mean(evaluation.data[method_name][f'sim']['pearson'])
        spearman = np.mean(evaluation.data[method_name][f'sim']['spearman'])
        atlas_diff = evaluation.data[method_name][f'sim']['atlas-diff']
        print(f'{pretty_names[i]} & {mse:.6f} & {chi2:.6f} & {pearson:.4f} & {spearman:.4f} & {atlas_diff:.6f} \\\\')

    print('Finished')


if __name__ == '__main__':
    for _ in range(50):
        main()
