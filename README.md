MetDecode
---

## 1) Correct the atlas for biases

```bash
python3 correct.py atlas.tsv cfdna.tsv corrected-atlas.tsv
```

Both input files `atlas.tsv` and `cfdna.tsv` should follow this format:
```
CHROM	START	END	BLCA_METH	BLCA_DEPTH	BRCA_METH	BRCA_DEPTH  ...
chr1	1161075	1161776	1491.000	1778	2052.000	2326    ...
chr1	1221864	1222225	955.000	1112	1176.000	1305    ...
chr1	1232963	1233492	2715.000	2904	3666.000	3933    ...
...
```
Values should be tab-separated, and the three first columns correspond to the chromosome,
start and end positions of the marker region, respectively. Starting from the fourth column, column `2i`
gives the methylated count for cell type `i`, and column `2i+1` the total count for cell type `i`.
Header should be of the same form as shown above with the first 3 columns being `CHROM`, `START` and `END`,
and the other column names ending either with `_METH` or `_DEPTH`.

Output file `corrected-atlas.tsv` will be saved in the exact same format.

The `correct.py` script has optional arguments:
- `-p`: The degree of importance attached to the coverage. `-p 0` enforces uniform weights on the methylation ratios, while `-p 1` re-weight them based on the read counts. Any positive value (e.g. `-p 0.4`) is allowed.
- `-lambda1`: Regularisation of the Gamma matrix. High values will constrain the Gamma matrix to stay close to the input methylation ratios.
- `-lambda2`: Regularisation of the bias terms. High values will constrain the corrected atlas to stay close to the Gamma matrix.
- `-max-correction`: Maximum difference allowed between the input methylation ratios and the corrected ratios.
- `-multiplicative`: Whether to perform multiplicative bias correction `sigma(sigma^{-1}(gamma) + u * v)` instead of additive correction `sigma(sigma^{-1}(gamma) + u + v)`.
- `-n-unknown-tissues`: Number of atlas entities to infer automatically from the cfDNA samples and add to the current atlas.
- `-n-hidden`: Relevant only when `-n-unknown-tissues` is strictly greater than 0. Number of hidden neurons within each layer of the neural network. Lower values prevent the network from overfitting the data and inferring atlas entities that correlate too much with the input cfDNA samples.
- `-maxit`: Maximum number of iterations. Reducing this number increases the speed but may produce less accurate results. 

## 2) Deconvolute the samples using the corrected atlas

```bash
python3 deconvolute.py corrected-atlas.tsv cfdna.tsv alpha.csv
```

Output file is a CSV file containing the estimated cell type contributions
from each atlas entity to each cfDNA sample.
Contributions are given as percentages and will always sum up to one.
