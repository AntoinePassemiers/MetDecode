# -*- coding: utf-8 -*-
#
#  run.py
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

import argparse
import os

import numpy as np

from metdecode.io import load_input_file
from metdecode.model import MetDecode
from metdecode.utils import bounded_float_type


parser = argparse.ArgumentParser()
parser.add_argument(
    'atlas-filepath',
    type=str,
    help='Location of the input atlas file (tsv file)'
)
parser.add_argument(
    'cfdna-filepath',
    type=str,
    help='Location of the input atlas file (tsv file)'
)
parser.add_argument(
    'out-filepath',
    type=str,
    help='Where to write the deconvolution results as CSV file (e.g. alpha.csv)'
)
parser.add_argument(
    '-n-unknown-tissues',
    type=int,
    default=0,
    help='Number of unknown tissues to infer and add to the atlas'
)
parser.add_argument(
    '-beta',
    type=bounded_float_type(lb=0),
    default=0.5,
    help='Importance attached to the coverage'
)
args = parser.parse_args()

# Input and output files
ATLAS_FILEPATH = getattr(args, 'atlas-filepath')
CFDNA_FILEPATH = getattr(args, 'cfdna-filepath')

# Create output folder if not exists
OUT_FILEPATH = getattr(args, 'out-filepath')
folder = os.path.dirname(OUT_FILEPATH)
if (folder != '') and (not os.path.isdir(os.path.dirname(OUT_FILEPATH))):
    os.makedirs(folder)

# Parse input files
M_atlas, D_atlas, cell_type_names, marker_names = load_input_file(ATLAS_FILEPATH)
M_cfdna, D_cfdna, sample_names, marker_names2 = load_input_file(CFDNA_FILEPATH)
for marker1, marker2 in zip(marker_names, marker_names2):
    marker1 = marker1.split('-')
    marker2 = marker2.split('-')
    if (marker1[0] != marker2[0]) or (marker1[1] != marker2[1]):
        raise ValueError(f'Marker regions differ in the two input files: {marker1} and {marker2}')

# Do a few checks
assert(np.less_equal(M_cfdna, D_cfdna).all())
assert(np.less_equal(M_atlas, D_atlas).all())
assert(M_cfdna.shape[1] == M_atlas.shape[1])
assert(M_cfdna.shape == D_cfdna.shape)
assert(M_atlas.shape == D_atlas.shape)

# Print summary
print(f'Input atlas file        : {ATLAS_FILEPATH}')
print(f'Input cfdna file        : {CFDNA_FILEPATH}')
print(f'Output file             : {OUT_FILEPATH}')
print('Number of cfDNA profiles       : %i' % M_cfdna.shape[0])
print('Number of tissues in the atlas : %i' % M_atlas.shape[0])
print('Number of markers              : %i' % M_atlas.shape[1])

# Deconvolution
model = MetDecode(M_atlas, D_atlas, M_cfdna, D_cfdna, n_unknown_tissues=args.n_unknown_tissues)
alpha = model.deconvolute()

# Results are stored in Alpha_hat,
# where Alpha_hat[i, j] is the contribution of tissue j
# in cfDNA profile i
with open(OUT_FILEPATH, 'w') as f:
    f.write(','.join(['Sample'] + list(cell_type_names) + [f'Unknown{i + 1}' for i in range(args.n_unknown_tissues)]) + '\n')
    for i, sample_name in enumerate(sample_names):
        f.write(sample_name)
        for value in alpha[i, :]:
            percentage = 100. * value
            f.write(f',{percentage:.3f}')
        f.write('\n')
print(f'Deconvolution results stored at {OUT_FILEPATH}.')