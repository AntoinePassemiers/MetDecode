from typing import Tuple, Optional

import numpy as np
import pandas as pd


def chr_name_to_int(chr_name: str) -> int:
    """Convert chromosome name to identifier.

    Args:
        chr_name: Chromosome name (e.g., "chr8").

    Returns:
        Integer representation of the chromosome.
    """
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


def row_name_as_int(combined_coords: str) -> int:
    """Convert genomic coordinates to integer.

    These integers are further used to sort marker regions
    by genomic coordinates efficiently.

    Args:
        combined_coords: Concatenation of genomic coordinates
            of the form "chr1-3792834-3793456".

    Returns:
        Integer representation of the marker region.
    """
    elements = combined_coords.split('-')
    assert len(elements) == 3
    chr_name = elements[0]
    start = int(elements[1])
    return 1000000000 * chr_name_to_int(chr_name) + start


def load_input_file(
        filepath: str,
        chr_idx: Optional[np.ndarray] = None,
        values_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load input TSV file containing the counts.

    Each tissue / cell type has two dedicated columns, namely the number of methylated CpG
    sites spanned in the marker region, and the total number of CpG sites
    (both methylation and unmethylated). Each row corresponds to a marker region.
    The first 3 columns contain respectively the chromosome, start position and end position
    of each marker region. The file must contain a header of the form:
    CHROM    START   END TISSUE1_METH    TISSUE1_DEPTH   TISSUE2_METH    ...

    Args:
        filepath: Input TSV file.

    Returns:
        methylated: A matrix of shape `(n_samples, n_markers)` containing the methylated counts
            for each sample and marker region.
        depths: A matrix of shape `(n_samples, n_markers)` containing the total counts
            (methylated + unmethylated) for each sample and marker region.
        column_names: List of column (marker) names.
        row_names: List of row (sample) names.
    """

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
    if chr_idx is None:
        row_names = df.iloc[:, 0].astype(str) + '-' + df.iloc[:, 1].astype(str) + '-' + df.iloc[:, 2].astype(str)
    else:
        row_names = ''
        for k, i in enumerate(chr_idx):
            row_names = row_names + df.iloc[:, i].astype(str)
            if k < len(chr_idx) - 1:
                row_names = row_names + '-'
    df = df.set_index(row_names, drop=False)

    # Retrieve the count matrices
    if values_idx is None:
        values = df.iloc[:, 3:].values.T
    else:
        values = df.iloc[:, values_idx].values.T
    df.reset_index()
    methylated = values[::2, :].astype(float)
    depths = values[1::2, :].astype(int)

    # Row and column labels
    if header is not None:
        if values_idx is None:
            column_names = [s.replace('_METH', '') for s in df.columns[3::2]]
        else:
            column_names = [s.replace('_METH', '') for s in df.columns[values_idx][::2]]
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
    """Save counts to TSV file.

    This function essentially performs the opposite of `load_input_file`.

    Args:
        filepath: Output TSV file.
        methylated: A matrix of shape `(n_samples, n_markers)` containing the methylated counts
            for each sample and marker region.
        depths: A matrix of shape `(n_samples, n_markers)` containing the total counts
            (methylated + unmethylated) for each sample and marker region.
        column_names: List of column (marker) names.
        row_names: List of row (sample) names.
    """
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
                # s += f'\t{float(methylated[i, j]):.3f}\t{int(depths[i, j])}'
                s += f'\t{methylated[i, j]}\t{int(depths[i, j])}'
            f.write(s + '\n')
