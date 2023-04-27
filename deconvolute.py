import argparse
import os

import numpy as np

from metdecode.core import MetDecode
from metdecode.io import load_input_file


# Argument parser
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
    '-p',
    type=float,
    default=0.97,
    help='Importance of coverage'
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

# Print summary
print(f'Input atlas file        : {ATLAS_FILEPATH}')
print(f'Input cfdna file        : {CFDNA_FILEPATH}')
print(f'Output file             : {OUT_FILEPATH}')

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

print('Number of cfDNA profiles       : %i' % M_cfdna.shape[0])
print('Number of tissues in the atlas : %i' % M_atlas.shape[0])
print('Number of markers              : %i' % M_atlas.shape[1])

model = MetDecode(p=args.p)
model.set_atlas(M_atlas, D_atlas)
alpha = model.deconvolute(M_cfdna, D_cfdna)

# Results are stored in Alpha_hat,
# where Alpha_hat[i, j] is the contribution of tissue j
# in cfDNA profile i
with open(OUT_FILEPATH, 'w') as f:
    f.write(','.join(['Sample'] + list(cell_type_names)) + '\n')
    for i, sample_name in enumerate(sample_names):
        f.write(sample_name)
        for value in alpha[i, :]:
            percentage = 100. * value
            f.write(f',{percentage:.3f}')
        f.write('\n')

print(f'Deconvolution results stored at {OUT_FILEPATH}.')
