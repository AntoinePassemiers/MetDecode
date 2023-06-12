import os

import hyperopt
import numpy as np
from scipy.stats import pearsonr

from metdecode.io import load_input_file
from metdecode.md2 import MetDecodeV2
from metdecode.model import Model

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, 'data')
ATLAS_FILEPATH = os.path.join(DATA_FOLDER, 'atlas_insil.tsv')

ADD_UNKNOWN = True
USE_MD2 = False


def compute_reg_loss(alpha: np.ndarray) -> float:

    loss = np.mean(np.maximum(0, 100 * alpha[:, -1] - 15)) + 5 * np.mean(np.maximum(0, 5 - 100 * alpha[:, -1]))

    # target_mu = np.asarray([0.05651396, 0.15084027, 0.06981952, 0.03278167, 0.08573908, 0.04916338, 0.54163744]) * 100
    # target_sigma = np.asarray([0.03092762, 0.04793517, 0.03413288, 0.02384947, 0.03749949, 0.02895841, 0.06673595]) * 100

    return loss


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

    assert len(cell_type_names) == len(M_atlas)
    assert len(cell_type_names) == len(D_atlas)

    if USE_MD2:
        model = MetDecodeV2(M_atlas,
            D_atlas,
            M_cfdna,
            D_cfdna,n_unknown_tissues=1, correction=True)
        alpha = model.deconvolute()
    else:
        model = Model(**params)
        alpha = model.fit(
            M_atlas,
            D_atlas,
            M_cfdna,
            D_cfdna,
            n_unknown_tissues=int(ADD_UNKNOWN)
        )

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

    unk_reg = compute_reg_loss(alpha)

    print(f'Average unknown contribution: {100 * np.mean(alpha[:, -1])}')

    print(f'MAE: {mae}')
    print(f'LOD: {lod}')
    print(f'CORR: {corr}')
    print(f'UNK_REG: {unk_reg}')

    score = lod + 0.05 * unk_reg

    return score


def evaluate_cfdna(params) -> float:
    cfdna_filepath = os.path.join(DATA_FOLDER, 'bothbatch.tims.txt')
    M_atlas, D_atlas, cell_type_names, _ = load_input_file(ATLAS_FILEPATH)
    M_atlas = M_atlas[:-1, :]
    D_atlas = D_atlas[:-1, :]
    cell_type_names = cell_type_names[:-1]
    M_cfdna, D_cfdna, sample_names, _ = load_input_file(cfdna_filepath)

    model = Model(**params)
    alpha = model.fit(
        M_atlas,
        D_atlas,
        M_cfdna,
        D_cfdna,
        n_unknown_tissues=int(ADD_UNKNOWN)
    )

    with open(f'alpha.csv', 'w') as f:
        f.write(','.join(['Sample'] + list(cell_type_names)) + '\n')
        for i, sample_name in enumerate(sample_names):
            f.write(sample_name)
            for value in alpha[i, :]:
                percentage = 100. * value
                f.write(f',{percentage:.3f}')
            f.write('\n')

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

    error = 0
    losses = []
    for i in range(len(labels)):
        if labels[i] > 0:
            if np.argmax(y_pred[:, i]) != labels[i] - 1:
                error += 1
            losses.append(np.mean(y_pred[:, i]))
        else:
            mask = np.ones(4, dtype=bool)
            mask[labels[i] - 1] = False
            losses.append(np.mean(y_pred[mask, i]) - y_pred[labels[i] - 1, i])
    loss = np.mean(losses)

    unk_reg = compute_reg_loss(alpha)

    print(f'Average unknown contribution: {100 * np.mean(alpha[:, -1])}')

    print(f'UNK_REG: {unk_reg}')
    print(f'Loss: {loss}')
    print(f'ERROR: {error}')

    score = loss + 0.05 * unk_reg

    return score


def objective(params):
    try:
        loss = evaluate_cfdna(params)
        losses = []
        # for dataset in ['ov33', 'br62', 'cer77', 'colo']:
        # for dataset in ['br62', 'cer77', 'colo', 'ov33', 'br66', 'cer81', 'colo45', 'ov79']:
        for dataset in ['cer77']:
            losses.append(evaluate(dataset, params))
        loss += float(np.mean(losses))
        print(f'Total loss: {loss}')
        return {'loss': loss, 'status': hyperopt.STATUS_OK}
    except Exception as e:
        print(e)
        raise e
        return {'loss': np.inf, 'status': hyperopt.STATUS_FAIL, 'status_fail': str(e)}


search_space = {
    'p': hyperopt.hp.uniform('p', 0, 1),
    'lts_ratio': hyperopt.hp.uniform('lts_ratio', 0, 1),
    'cta_ratio': hyperopt.hp.uniform('cta_ratio', 0, 1),
    'cta_type': hyperopt.hp.choice('cta_type', ['perm', 'ot']),
    'lambda1': hyperopt.hp.loguniform('lambda1', -1, 2),
    'lambda2': hyperopt.hp.loguniform('lambda2', -1, 2),
    'obs_criterion': hyperopt.hp.choice('obs_criterion', ['l1', 'l2', 'eps-svr']),
    'ref_criterion': hyperopt.hp.choice('ref_criterion', ['l1', 'l2', 'eps-svr']),
    'obs_criterion_eps': hyperopt.hp.loguniform('obs_criterion_eps', -5, 0),
    'ref_criterion_eps': hyperopt.hp.loguniform('ref_criterion_eps', -5, 0),
    'obs_criterion_nu': hyperopt.hp.uniform('obs_criterion_nu', 0, 1),
    'ref_criterion_nu': hyperopt.hp.uniform('ref_criterion_nu', 0, 1),
}

"""
        losses = []
best_params = hyperopt.fmin(
    fn=objective,
    space=search_space,
    algo=hyperopt.tpe.suggest,
    max_evals=1000
)
"""

#search_space = SearchSpace(15, **search_space)
#opt = EvolutionaryOptimizer(pop_size=20, partition_size=5, n_iter=10000)
#opt.run(objective, search_space)

objective({})

print('Finished')
