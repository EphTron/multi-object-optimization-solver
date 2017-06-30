#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 30.06.17 17:02
@author: ephtron
"""

import matplotlib.pyplot as plt


def plot_generation_boxplot(population):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # get fitness data array of population
    data = [candidate.get_fitness() for candidate in population]

    # Create the boxplot
    bp = ax.boxplot(data)
    plt.show()


def get_candidate_vector(solution_candidate):
    vec = []
    for value in solution_candidate.get_feature_dict().values():
        if value is None:
            vec.append(0)
        else:
            vec.append(1)

    return vec
