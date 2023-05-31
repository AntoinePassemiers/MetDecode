# -*- coding: utf-8 -*-
#
#  evolutionary.py
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

import json
import random
from typing import Dict, Callable

import hyperopt.pyll
import numpy as np
from numpyencoder import NumpyEncoder


class Individual:

    def __init__(self, **kwargs):
        self.params: Dict = kwargs

    def mutate_inplace(self, search_space: 'SearchSpace'):
        key = random.choice(list(self.params.keys()))
        self.params[key] = hyperopt.pyll.stochastic.sample(search_space.params[key])

    def cross_over(self, other: 'Individual') -> 'Individual':
        params = {}
        for key in self.params.keys():
            params[key] = self.params[key] if (random.random() < 0.5) else other.params[key]
        return Individual(**params)

    def as_dict(self) -> Dict:
        return {
            **self.params
        }


class SearchSpace:

    def __init__(self, n_tissues: int, **kwargs):
        self.n_tissues: int = n_tissues
        self.params: Dict = kwargs

    def random(self) -> Individual:
        values = {}
        for param_name, param in self.params.items():
            values[param_name] = hyperopt.pyll.stochastic.sample(param)
        return Individual(**values)


class EvolutionaryOptimizer:

    def __init__(self, pop_size=10, n_iter=1000, partition_size=5,
                 mutation_rate=0.5, mutation_std=0.3, early_stopping=300):
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.partition_size = partition_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.early_stopping = early_stopping
        self.scores = []

    def new_sol(self, pop, scores, search_space):
        # Shuffle the population
        indices = np.arange(len(pop))
        np.random.shuffle(indices)
        scores = scores[indices]

        # Elect a winner in each of the two partitions
        ps = self.partition_size
        left_winner = pop[indices[np.argmax(scores[:ps])]]
        right_winner = pop[indices[ps + np.argmax(scores[ps:2*ps])]]

        # Apply the cross-over and mutation operators
        individual = left_winner.cross_over(right_winner)
        individual.mutate_inplace(search_space)
        return individual

    def run(self, func: Callable, search_space: SearchSpace, verbose=True):

        def obj(ind: Individual) -> float:
            loss = func(ind.as_dict())
            with open('hp-results.txt', 'a') as f:
                f.write(json.dumps({
                    'loss': loss,
                    'params': ind.params,
                }, cls=NumpyEncoder) + '\n')
            return -float(loss['loss'])

        # Initialize population
        pop = [search_space.random() for _ in range(self.pop_size)]

        # Set initial solution as the best one so far
        self.scores = []
        best_score = -np.inf
        best_iteration = 0

        # Compute fitness functions on all individuals
        scores = np.asarray([obj(ind) for ind in pop], dtype=float)

        for k in range(self.n_iter):

            # Create new solution to replace worst solution
            new_ind = self.new_sol(pop, scores, search_space)
            worst = np.argmin(scores)
            pop[worst] = new_ind
            scores[worst] = obj(new_ind)
            assert(not np.isnan(scores[worst]))

            # Check if improvement
            if scores[worst] > best_score:
                best_score = scores[worst]
                best_iteration = k
            if verbose and (k + 1) % 100 == 0:
                print('Log-likelihood at iteration %i: %f' \
                    % (k + 1, best_score))
            self.scores.append(best_score)

            if np.isnan(best_score):
                if verbose:
                    print('[Warning] Invalid value encountered in heuristic solver')
                break

            # Stop algorithm if no more improvement
            if k - best_iteration >= self.early_stopping:
                break
