#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser


def assess_fitness(features, interactions):
    feature_fitness = 0
    for feature in features:
        feature_fitness += feature.value

    interaction_fitness = 0
    for i in interactions:
        interaction_fitness += i.get_value(features)

    return feature_fitness + interaction_fitness


if __name__ == "__main__":
    features, interactions = feature_parser.parse('src/project_public_1/bdbc', verbose=False)
