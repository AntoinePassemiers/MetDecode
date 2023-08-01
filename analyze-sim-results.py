import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

ROOT = os.path.dirname(os.path.abspath(__file__))

EXP1 = False

# METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nu', 'metdecode2-nc', 'metdecode2']
METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nc' if not EXP1 else 'metdecode2-nu']

res = []
for filename in os.listdir(ROOT):

    if not (filename.startswith('results2-' if not EXP1 else 'results-') and filename.endswith('.pkl')):
        continue
    filepath = os.path.join(ROOT, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    data = data[0]
    res.append([data[method_name]['sim']['pearson'] for method_name in METHOD_NAMES])
res = np.asarray(res)

print(res.shape)

colors = ['darkblue', 'royalblue', 'darkslateblue', 'slateblue', 'mediumvioletred', 'palevioletred', 'steelblue', 'darkturquoise', 'darkcyan', 'mediumseagreen', 'darkgreen', 'green', 'yellowgreen', 'tan']
pretty_names = [
    'BRCA', 'CEAD', 'CESC', 'COAD', 'OV', 'READ', 'B cell', 'CD4+ T-cell', 'CD8+ T-cell',
    'Erythroblast', 'Monocyte', 'Natural killer cell', 'Neutrophil', 'Average']
plt.figure(figsize=(12.5, 7.5))
for k in range(14):
    ax = plt.subplot(3, 5, k + 1)
    if k == 13:
        ys = list(np.mean(res, axis=2).T)
    else:
        ys = list(res[:, :, k].T)

    p_value = ttest_rel(ys[0], ys[1], alternative='less').pvalue

    r = plt.violinplot(ys, showmeans=True, showextrema=True)
    color = colors[k]
    r['cbars'].set_color(color)
    r['cmins'].set_color(color)
    r['cmaxes'].set_color(color)
    r['cmeans'].set_color(color)
    for body in r['bodies']:
        body.set_color(color)
    plt.title(f'{pretty_names[k]} ({p_value:.3f})')
    plt.grid(alpha=0.4, color='grey', linewidth=0.5, linestyle='--')
    plt.xticks([1, 2], ['beta=0', 'beta=1'] if EXP1 else ['unk=0', 'unk=1'])
    ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plt.savefig('sim1.png' if EXP1 else 'sim2.png', dpi=400)
plt.show()
