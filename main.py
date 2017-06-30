#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser
import candidate_solution
import utility

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
    population = [candidate_solution.generate_random(features) for i in range(0,50)]

    for candidate in population:
        print(utility.get_candidate_vector(candidate))

    # get array with all fitness values
    fitness_data = [assess_fitness(candidate, interactions) for candidate in population]

    utility.plot_generation_boxplot(fitness_data)
    print("Max Fitness:", max(fitness_data), "\nFitness Data:", fitness_data)
    

if __name__ == "__main__":
    evolution('src/project_public_1/bdbc', verbose=False)
