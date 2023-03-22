# -*- coding: utf-8 -*-
#
#  io.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
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

from typing import Tuple

import numpy as np
import pandas as pd


def chr_name_to_int(chr_name: str) -> int:
    chr_name = chr_name.lower()
    if chr_name.startswith('chr'):
        chr_name = chr_name[3:]
        if chr_name == 'x':
            chr_id = 23
        elif chr_name == 'y':
            chr_id = 24
        else:
            chr_id = int(chr_name)
    else:
        chr_id = int(chr_name)
    return chr_id - 1


def row_name_as_int(row_name: str) -> int:
    elements = row_name.split('-')
    assert len(elements) == 3
    chr_name = elements[0]
    start = int(elements[1])
    return 1000000000 * chr_name_to_int(chr_name) + start


def load_input_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Check the presence of the header
    with open(filepath, 'r') as f:
        line = f.readline()
        if line.startswith('CHROM'):
            header = 'infer'
        else:
            header = None

    # Parse input file
    df = pd.read_csv(filepath, delimiter='\t', header=header)

    # Reindex profiles by genomic coordinates
    row_names = df.iloc[:, 0].astype(str) + '-' + df.iloc[:, 1].astype(str) + '-' + df.iloc[:, 2].astype(str)
    df = df.set_index(row_names, drop=False)

    # Retrieve the count matrices
    values = df.iloc[:, 3:].values.T
    df.reset_index()
    methylated = values[::2, :].astype(float)
    depths = values[1::2, :].astype(int)

    # Row and column labels
    if header is not None:
        column_names = [s.replace('_METH', '') for s in df.columns[3::2]]
    else:
        column_names = [f'sample-{j + 1}' for j in range(depths.shape[0])]
    column_names = np.asarray(column_names, dtype=object)
    row_names = np.asarray(row_names, dtype=object)

    # Sort marker regions by increasing genomic coordinates
    idx = np.argsort(np.asarray([row_name_as_int(row_name) for row_name in row_names], dtype=np.uint64))
    row_names = row_names[idx]
    methylated = methylated[:, idx]
    depths = depths[:, idx]

    # Ensure methylation ratios are within bounds [0, 1]
    methylated = np.clip(methylated, 0, depths)

    return methylated, depths, column_names, row_names


def save_counts(
        filepath: str,
        methylated: np.ndarray,
        depths: np.ndarray,
        column_names: np.ndarray,
        row_names: np.ndarray
):
    n_samples = methylated.shape[0]
    n_markers = methylated.shape[1]
    with open(filepath, 'w') as f:

        # Write header
        header = ['CHROM', 'START', 'END']
        for column_name in list(column_names):
            header.append(f'{column_name}_METH')
            header.append(f'{column_name}_DEPTH')
        f.write('\t'.join(header) + '\n')

        # Write lines
        for j in range(n_markers):
            elements = row_names[j].split('-')
            assert len(elements) == 3
            s = '\t'.join(elements)
            for i in range(n_samples):
                s += f'\t{float(methylated[i, j]):.3f}\t{int(depths[i, j])}'
            f.write(s + '\n')
