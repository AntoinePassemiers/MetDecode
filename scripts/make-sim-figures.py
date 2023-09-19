# -*- coding: utf-8 -*-
#
#  make-sim-figures.py
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
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_FOLDER = os.path.join(ROOT, 'sim-results')

for EXP1 in [True, False]:

    METHOD_NAMES = ['metdecode2-nc-nu', 'metdecode2-nu' if not EXP1 else 'metdecode2-nc']

    res = []
    if EXP1:
        folder = 'unknowns'
    else:
        folder = 'coverage'
    for filename in os.listdir(os.path.join(OUT_FOLDER, folder)):

        if not (filename.startswith('results-') and filename.endswith('.pkl')):
            continue
        filepath = os.path.join(OUT_FOLDER, folder, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        data = data[0]
        print(filepath)
        print(data.keys())
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

        p_value = ttest_rel(ys[0], ys[1]).pvalue

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
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig('sim1.png' if EXP1 else 'sim2.png', dpi=400)
    plt.close()
    plt.clf()
    plt.cla()
