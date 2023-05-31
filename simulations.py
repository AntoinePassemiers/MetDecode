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

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from metdecode.io import load_input_file
from scipy.special import expit

from metdecode.evaluation import Evaluation
from metdecode.model import Model


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas_insil.tsv')


def wgbs_var_func(x):
    a = -33.26707218
    b = 68.81917451
    c = -48.50627055
    d = 13.16493979
    e = -6.96673189
    y_hat = np.exp(a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e)
    return np.clip(y_hat, 0.001, x * (1. - x))


def wgbs_mean_func(x):
    a = 9.18240158e+02
    b = -3.10896889e+03
    c = 4.24023626e+03
    d = -2.98356794e+03
    e = 1.16071968e+03
    f = -2.52894933e+02
    g = 3.43328033e+01
    h = -3.62453197e+00
    tmp = a * x ** 7 + b * x ** 6
    tmp += c * x ** 5 + d * x ** 4 + e * x ** 3 + f * x ** 2 + g * x + h
    y_hat = expit(tmp)
    return np.clip(y_hat, 0.0001, 0.9999)


def wgbs_to_em(gamma, bias_func, noisy=True):
    mean = bias_func(gamma)
    X_clean = mean
    if noisy:
        sfs = np.random.rand(1, gamma.shape[1])
        var = wgbs_var_func(gamma)
        var = np.clip(var, 0.001, 0.999 * mean * (1. - mean))
        tmp = ((mean * (1. - mean)) / var - 1.)
        alpha = mean * tmp
        beta = (1. - mean) * tmp
        X_noisy = scipy.stats.beta.isf(sfs, alpha, beta)
        # X_noisy = X_noisy * 0.5 + gamma * 0.5  # TODO
    else:
        X_noisy = X_clean
    return X_noisy, X_clean


def generate_mix_dataset(n_profiles: int = 48, n_tissues: int = 13, n_known_tissues: int = 13,
                         n_markers: int = 4441,
                         bias: bool = True, verbose: bool = False, **kwargs) -> dict:

    M_cfdna, D_cfdna, sample_names, _ = load_input_file(os.path.join(DATA_FOLDER, 'bothbatch.tims.txt'))
    M_atlas, D_atlas, cell_type_names, _ = load_input_file(ATLAS_FILEPATH)
    print(cell_type_names)
    assert len(cell_type_names) == len(M_atlas)
    assert len(cell_type_names) == len(D_atlas)

    idx = np.arange(D_cfdna.shape[1])
    np.random.shuffle(idx)
    idx = idx[:n_markers]
    M_cfdna = M_cfdna[:, idx]
    D_cfdna = D_cfdna[:, idx]
    M_atlas = M_atlas[:, idx]
    D_atlas = D_atlas[:, idx]

    print(cell_type_names, len(cell_type_names))

    #assert R_depths.shape[0] == n_tissues
    #n_tissues = R_depths.shape[0]
    assert n_known_tissues <= n_tissues
    assert n_markers <= D_atlas.shape[1]

    R_depths = D_atlas + 1
    x_depths = np.round(np.mean(D_cfdna + 1, axis=0)[np.newaxis, :]).astype(int)
    X_depths = np.repeat(x_depths, n_profiles, axis=0).astype(int)
    R_methylated_orig = M_atlas + 1

    gamma = R_methylated_orig / R_depths
    alpha = np.zeros((n_profiles, n_tissues))
    for i in range(len(alpha)):
        alpha_ = np.random.dirichlet(np.asarray(
            [0.7393, 3.0938, 8.2576, 3.8222, 1.7946, 4.6937, 2.6914, 29.6514]))
        j = np.random.randint(0, 6)
        alpha[i, j] = alpha_[0]
        alpha[i, 6:] = alpha_[1:]
    cancer_mask = np.zeros(n_tissues, dtype=bool)
    cancer_mask[:6] = True
    print(np.sum(alpha, axis=1))

    # Add bias to atlas
    if bias:
        bias_func = wgbs_mean_func
        gamma_distorted, gamma_prime_clean = wgbs_to_em(gamma, bias_func, noisy=True)
        diff = gamma_distorted - gamma
        for j in range(diff.shape[0]):
            print(f'Expected difference in average methylation for atlas entity {j}: '
                  f'{np.mean(diff[j, :]) * 100.} % '
                  f'({np.mean(gamma[j, :]) * 100.} -> {np.mean(gamma_distorted[j, :]) * 100.})')
    else:
        gamma_distorted = gamma
        gamma_prime_clean = gamma

    coverage_factor = 1
    R_methylated = np.random.binomial(R_depths * coverage_factor, gamma_distorted)
    X_methylated = np.random.binomial(X_depths * coverage_factor, np.dot(alpha, gamma), size=(n_profiles, R_depths.shape[1]))
    R_methylated = np.round(R_methylated / coverage_factor).astype(int)
    X_methylated = np.round(X_methylated / coverage_factor).astype(int)
    # R_methylated = np.round(R_depths * gamma_distorted).astype(int)
    # X_methylated = np.round(X_depths * np.dot(alpha, gamma)).astype(int)

    assert R_methylated.shape[1] == n_markers
    assert R_depths.shape[1] == n_markers

    is_female = np.zeros(n_profiles, dtype=bool)
    age = np.full(n_profiles, 20)
    enzymatic = np.ones(n_profiles, dtype=bool)

    R_methylated = R_methylated[:n_known_tissues, :]
    R_depths = R_depths[:n_known_tissues, :]

    assert alpha.shape == (n_profiles, n_tissues)
    assert gamma.shape == (n_tissues, n_markers)
    assert X_methylated.shape == (n_profiles, n_markers)
    assert X_depths.shape == (n_profiles, n_markers)
    assert R_methylated.shape == (n_known_tissues, n_markers)
    assert R_depths.shape == (n_known_tissues, n_markers)

    return {
        'X-methylated': X_methylated,
        'X-depths': X_depths,
        'R-methylated': R_methylated,
        'R-depths': R_depths,
        'Alpha': alpha,
        'Gamma': gamma,
        'Gamma-prime-clean': gamma_prime_clean,
        'Gamma-prime-noisy': gamma_distorted,
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

    METHOD_NAMES = ['metdecode2-nocorrection', 'metdecode2']

    for method_name in METHOD_NAMES:

        print(f'Running method "{method_name}"')

        gamma_hat = dataset['Gamma-prime-noisy']
        if method_name == 'metdecode2-nocorrection':
            model = Model()
            Alpha_hat = model.fit(*args, n_unknown_tissues=1, infer=False)
            gamma_hat = model.R_atlas
        elif method_name == 'metdecode2':
            model = Model()
            Alpha_hat = model.fit(*args, n_unknown_tissues=1, infer=True)
            gamma_hat = model.R_atlas
        else:
            raise NotImplementedError(f'Unknown method "{method_name}"')

        print(Alpha_hat.shape, dataset['Alpha'].shape, dataset['Gamma'].shape, dataset['X-methylated'].shape, dataset['X-depths'].shape)

        evaluation.add(Alpha_hat, dataset['Alpha'], gamma_hat, dataset['Gamma'],
            dataset['X-methylated'], dataset['X-depths'], method_name, exp_id, n_known_tissues)


if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUT_FOLDER = os.path.join(ROOT, 'results')
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    N_REPEATS = 1

    dataset = generate_mix_dataset(
        bias=True,
        n_profiles=200,
        n_markers=1000
    )
    evaluation = Evaluation()
    evaluate(dataset, evaluation, f'sim')
    evaluation.save('results-sim.pkl')
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    colors = [
        'slateblue', 'mediumvioletred', 'goldenrod', 'darkseagreen'
    ]
    # pretty_names = ['NNLS', 'CelFiE', 'MetDecode']
    pretty_names = ['MetDecode (no correction)', 'MetDecode']

    # methods = ['nnls', 'celfie', 'metdecodeV4']
    methods = ['metdecode2-nocorrection', 'metdecode2']

    cell_type_names = [
        'BRCA', 'CEAD+CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T-cell', 'CD8+ T-cell',
        'Erythroblast', 'Monocyte', 'Natural killer cell', 'Neutrophil']
    fig = plt.figure(figsize=(12, 9))
    for j in range(12):
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

    """
    evaluation = Evaluation.load(os.path.join(OUT_FOLDER, 'results-sim.pkl'))
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    gamma = dataset['Gamma']
    gamma_prime_clean = dataset['Gamma-prime-clean']
    gamma_distorted = dataset['Gamma-prime-noisy']
    gamma_pred = evaluation.data['metdecode2'][f'sim']['gamma-pred']

    d = (gamma_prime_clean - gamma)
    d_pred = (gamma_pred - gamma)
    plt.plot([-1, 1], [-1, 1])
    plt.scatter(d.flatten(), d_pred.flatten(), alpha=0.1, color='royalblue')
    plt.xlabel('Meth. difference between EMseq and bisulfite atlas')
    plt.ylabel('Bias estimated by MetDecode')
    plt.show()
    """

    print('Finished')
