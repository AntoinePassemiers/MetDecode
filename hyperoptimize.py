import json
import os
from typing import Tuple

import hyperopt
import numpy as np
from numpyencoder import NumpyEncoder
from scipy.stats import pearsonr

from metdecode.io import load_input_file
from metdecode.core import MetDecode

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas_insil.tsv')

ADD_UNKNOWN = True


def evaluate(dataset: str, params) -> Tuple[float, float, float]:
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

    model = MetDecode(**params)
    model.fit(
        M_atlas,
        D_atlas,
        M_cfdna,
        D_cfdna,
        n_unknown_tissues=int(ADD_UNKNOWN),
        max_n_iter=2000
    )
    alpha = model.deconvolute(M_cfdna, D_cfdna)

    def compute_average_lod(y_pred, y_target, target_k):
        lods = []
        for i in range(y_pred.shape[1]):
            idx = np.argmax(y_pred[:, i, :], axis=0)
            lod = 100
            for j in range(len(idx)):
                if idx[j] == target_k:
                    lod = y_target[i, j]
                else:
                    break
            lods.append(lod)
        return np.mean(lods)


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

    print(f'MAE: {mae}')
    print(f'LOD: {lod}')
    print(f'CORR: {corr}')

    return mae, lod, corr


def objective(params):
    losses = []
    for dataset in ['br62', 'cer77', 'colo', 'ov33']:
        mae, lod, _ = evaluate(dataset, params)
        losses.append(lod)
    loss = float(np.mean(losses))
    with open('hp-results-unk1-new.txt', 'a') as f:
        f.write(json.dumps({'loss': loss, 'params': params}, cls=NumpyEncoder) + '\n')
    return {'loss': loss, 'status': hyperopt.STATUS_OK}


search_space = {
    'max_correction': hyperopt.hp.uniform('max_correction', 0, 1),
    'p': hyperopt.hp.uniform('p', 0, 2),
    'lambda1': hyperopt.hp.loguniform('lambda1', -5, 5),
    'lambda2': hyperopt.hp.loguniform('lambda2', -5, 5),
    'coverage_rcw': hyperopt.hp.choice('coverage_rcw', [False, True]),
    'multiplicative': hyperopt.hp.choice('multiplicative', [False, True]),
}
if ADD_UNKNOWN:
    search_space['n_hidden'] = hyperopt.hp.randint('n_hidden', 9)

best_params = hyperopt.fmin(
    fn=objective,
    space=search_space,
    algo=hyperopt.tpe.suggest,
    max_evals=1000
)

"""
lods, corrs = [], []
for dataset in ['br62', 'cer77', 'colo', 'ov33', 'br66', 'cer81', 'colo45', 'ov79']:
    _, lod, corr, = evaluate(dataset, {})
    lods.append(lod)
    corrs.append(corr)
    print('LODS', lods)
    print('CORRS', corrs)
"""
