# -*- coding: utf-8 -*-
#
#  cross-validation.py
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

import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from metdecode.core import MetDecode
from metdecode.io import load_input_file


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, '..', 'data')
OUT_FILEPATH = os.path.join(ROOT, '..', 'results')

ATLAS_CORRECTION = True

# Parse input files
M_atlas, D_atlas, entity_names, marker_names = load_input_file(os.path.join(DATA_DIR, 'atlas.tsv'))
M_cfdna, D_cfdna, sample_names, marker_names2 = load_input_file(os.path.join(DATA_DIR, 'cfdna.tsv'))
for marker1, marker2 in zip(marker_names, marker_names2):
    marker1 = marker1.split('-')
    marker2 = marker2.split('-')
    if (marker1[0] != marker2[0]) or (marker1[1] != marker2[1]):
        raise ValueError(f'Marker regions differ in the two input files: {marker1} and {marker2}')

# Create output folder if not exists
if not os.path.isdir(OUT_FILEPATH):
    os.makedirs(OUT_FILEPATH)

print(sample_names)

# Leave-one-out cross-validation
alpha = []
for i, (train_index, test_index) in tqdm.tqdm(enumerate(LeaveOneOut().split(M_cfdna))):
    test_index = int(test_index[0])
    filepath = os.path.join(OUT_FILEPATH, f'{sample_names[test_index]}.npz')

    if ATLAS_CORRECTION:
        if not os.path.exists(filepath):
            model = MetDecode()
            model.fit(M_atlas, D_atlas, M_cfdna[train_index, :], D_cfdna[train_index, :], max_n_iter=2000)
            contributions = np.squeeze(model.deconvolute(M_cfdna[test_index, np.newaxis, :], D_cfdna[test_index, np.newaxis, :]))

            # Save results
            np.savez(filepath, R=model.R_atlas, alpha=contributions)
        else:
            data = np.load(filepath)
            contributions = data['alpha']
    else:
        model = MetDecode()
        model.set_atlas(M_atlas, D_atlas)
        contributions = np.squeeze(model.deconvolute(M_cfdna[test_index, np.newaxis, :], D_cfdna[test_index, np.newaxis, :]))

    alpha.append(contributions)
alpha = np.asarray(alpha)

settings = [
    ([0], 'Breast'),
    ([4], 'Ovarian'),
    ([3, 5], 'Colorectal'),
    ([1, 2], 'Cervical')
]

for tissue_idx, cancer_name in settings:
    mask = np.asarray([sample_name.split('-')[0] in {'Control', cancer_name} for sample_name in sample_names], dtype=bool)
    y_hat = np.sum(alpha[:, tissue_idx], axis=1)
    y = np.asarray([sample_name.split('-')[0] == cancer_name for sample_name in sample_names], dtype=int)

    y_hat, y = y_hat[mask], y[mask]

    print(cancer_name, roc_auc_score(y, y_hat))
