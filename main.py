#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser
import candidate_solution
from candidate_solution import CandidateSolution

def assess_fitness(candidate, interactions):
    features = candidate.get_feature_list()
    feature_fitness = 0
    for feature in features:
        feature_fitness += feature.value

    interaction_fitness = 0
    for i in interactions:
        interaction_fitness += i.get_value(features)

    return feature_fitness + interaction_fitness

def evolution(file_name, verbose):
    features, interactions = feature_parser.parse(file_name, verbose=verbose)
    C = [candidate_solution.generate_random(features) for i in range(0,50)]
    for c in C:
        print ("fitness:", assess_fitness(c, interactions))
    

if __name__ == "__main__":
    evolution('src/project_public_1/bdbc', verbose=False)
