import json
import os
from typing import Tuple

import hyperopt
import numpy as np
from matplotlib import pyplot as plt
from numpyencoder import NumpyEncoder
from scipy.stats import pearsonr

from metdecode.io import load_input_file
from metdecode.core import MetDecode

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas_insil.tsv')

ADD_UNKNOWN = True
MAX_N_ITER = 2000

def evaluate(dataset: str, params) -> float:
    reverse = False
    if dataset == 'br62':
        target_k = 0
        purity = 0.7811
        reverse = True
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_br62.txt')
    elif dataset == 'br66':
        target_k = 0
        purity = 0.89045
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_br66.txt')
    elif dataset == 'cer77':
        target_k = 1
        purity = 0.44262
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_cer77.txt')
    elif dataset == 'cer81':
        target_k = 1
        purity = 0.42343
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_cer81.txt')
    elif dataset == 'colo45':
        target_k = 2
        purity = 0.8924
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_colo45.txt')
    elif dataset == 'colo':
        target_k = 2
        purity = 0.61173
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_colo.txt')
    elif dataset == 'ov33':
        target_k = 3
        purity = 0.8924
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_ov33.txt')
    elif dataset == 'ov79':
        target_k = 3
        purity = 0.22943
        cfdna_filepath = os.path.join(DATA_FOLDER, 'insil120_ov79.txt')
    else:
        raise NotImplementedError(f'Unknown dataset "{dataset}"')

    M_atlas, D_atlas, cell_type_names, _ = load_input_file(ATLAS_FILEPATH)
    M_cfdna, D_cfdna, sample_names, _ = load_input_file(cfdna_filepath)

    model = MetDecode(verbose=False, **params)
    model.fit(
        M_atlas,
        D_atlas,
        M_cfdna,
        D_cfdna,
        n_unknown_tissues=int(ADD_UNKNOWN),
        max_n_iter=MAX_N_ITER
    )
    alpha = model.deconvolute(M_cfdna, D_cfdna)

    def compute_average_lod(y_pred, y_target, target_k):
        lods = []
        for i in range(y_pred.shape[1]):
            idx = np.argmax(y_pred[:, i, :], axis=0)
            lod = 100
            for j in range(len(idx)):
                if y_pred[idx[j], i, j] == 0:
                    break
                if idx[j] == target_k:
                    lod = y_target[i, j]
                else:
                    break
            lods.append(lod)
        return np.mean(lods)

    with open(f'alpha-{dataset}.csv', 'w') as f:
        f.write(','.join(['Sample'] + list(cell_type_names)) + '\n')
        for i, sample_name in enumerate(sample_names):
            f.write(sample_name)
            for value in alpha[i, :]:
                percentage = 100. * value
                f.write(f',{percentage:.3f}')
            f.write('\n')

    y_pred = np.asarray([
        alpha[:, 0],
        alpha[:, 1] + alpha[:, 2],
        alpha[:, 3] + alpha[:, 5],
        alpha[:, 4]
    ]) * 100
    y_pred = y_pred.reshape(4, 10, 12)
    if reverse:
        y_pred = y_pred[:, :, ::-1]

    y_target = np.asarray([0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 40, 50][::-1] * 10)
    y_target = y_target.reshape(10, 12)
    y_target_corrected = y_target * purity

    mae = float(np.mean(np.abs(y_target_corrected - y_pred[target_k, ...])))
    lod = float(compute_average_lod(y_pred, y_target_corrected, target_k))
    corr = float(pearsonr(y_pred[target_k, ...].flatten(), y_target_corrected.flatten())[0])
    unk_reg = 2 * np.mean(np.maximum(0, 100 * alpha[:, -1] - 15)) + 10 * np.mean(np.maximum(0, 5 - 100 * alpha[:, -1]))


    print(f'Average unknown contribution: {100 * np.mean(alpha[:, -1])}')

    print(f'MAE: {mae}')
    print(f'LOD: {lod}')
    print(f'CORR: {corr}')
    print(f'UNK_REG: {unk_reg}')

    score = lod + 0.2 * unk_reg

    return score


def evaluate_cfdna(params) -> float:
    cfdna_filepath = os.path.join(DATA_FOLDER, 'bothbatch.tims.txt')
    M_atlas, D_atlas, cell_type_names, _ = load_input_file(ATLAS_FILEPATH)
    M_cfdna, D_cfdna, sample_names, _ = load_input_file(cfdna_filepath)

    model = MetDecode(verbose=False, **params)
    model.fit(
        M_atlas,
        D_atlas,
        M_cfdna,
        D_cfdna,
        n_unknown_tissues=int(ADD_UNKNOWN),
        max_n_iter=MAX_N_ITER
    )
    alpha = model.deconvolute(M_cfdna, D_cfdna)

    y_pred = np.asarray([
        alpha[:, 0],  # BRCA
        alpha[:, 1] + alpha[:, 2],  # CER
        alpha[:, 3] + alpha[:, 5],  # COLOR
        alpha[:, 4]  # OV
    ]) * 100

    labels = np.zeros(y_pred.shape[1], dtype=int)
    labels[40:62] = 1
    labels[62:69] = 3
    labels[69:71] = 2
    labels[71:78] = 4
    labels[124:146] = 1
    labels[146:153] = 3
    labels[153:155] = 2
    labels[155:162] = 4

    mask = (np.argmax(y_pred, axis=0) != labels - 1)
    loss = np.mean(np.max(y_pred, axis=0) * mask)

    unk_reg = 2 * np.mean(np.maximum(0, 100 * alpha[:, -1] - 15)) + 10 * np.mean(np.maximum(0, 5 - 100 * alpha[:, -1]))

    print(f'Average unknown contribution: {100 * np.mean(alpha[:, -1])}')

    print(f'UNK_REG: {unk_reg}')
    print(f'Loss: {loss}')

    score = loss + 0.2 * unk_reg

    return score


def objective(params):
    try:
        losses = []
        for dataset in ['ov33', 'br62', 'cer77', 'colo']:
        # for dataset in ['br62', 'br66', 'cer77', 'cer81', 'colo45', 'colo', 'ov33', 'ov79']:
            losses.append(evaluate(dataset, params))
        loss = float(np.mean(losses)) + evaluate_cfdna(params)
        with open('hp-results.txt', 'a') as f:
            f.write(json.dumps({'loss': loss, 'params': params}, cls=NumpyEncoder) + '\n')
        print(f'Total loss: {loss}')
        return {'loss': loss, 'status': hyperopt.STATUS_OK}
    except Exception as e:
        return {'loss': np.inf, 'status': hyperopt.STATUS_FAIL, 'status_fail': str(e)}


search_space = {
    'max_correction': hyperopt.hp.uniform('max_correction', 0.005, 1),
    'p': hyperopt.hp.uniform('p', 0, 1),
    'lambda1': hyperopt.hp.loguniform('lambda1', 0, 3),
    'budget': hyperopt.hp.uniform('budget', 0.0, 0.02),
    #'unk_clip': hyperopt.hp.choice('unk_clip', [False, True]),
    #'unk_qt': hyperopt.hp.choice('unk_qt', [False, True]),
    #'unk_bound': hyperopt.hp.choice('unk_bound', [False, True]),
    'unk_q': hyperopt.hp.uniform('unk_q', 0.0, 0.05),
}

"""
best_params = hyperopt.fmin(
    fn=objective,
    space=search_space,
    algo=hyperopt.tpe.suggest,
    max_evals=1000
)
"""

objective({})
