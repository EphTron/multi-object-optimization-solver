#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 30.06.17 17:02
@author: ephtron
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_generation_boxplot(ax, population, generation):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # get fitness data array of population
    data = np.asarray([candidate.get_fitness() for candidate in population])

    ax.boxplot(data, positions=[generation])

    ax.set_xlim(-0.5, generation+0.5)



def get_candidate_vector(solution_candidate):
    vec = []
    for value in solution_candidate.get_feature_dict().values():
        if value is None:
            vec.append(0)
        else:
            vec.append(1)

    return vec
