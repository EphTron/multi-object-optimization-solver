#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""
import time

import feature_parser
import candidate_solution
from candidate_solution import CandidateSolution
import csp_solver
from csp_solver import CSPSolver
import json_helper

import numpy as np
import matplotlib.pyplot as plt

import random

FIGURE_NAME = "Title"
FEATURE_PATHS = []
INTERACTION_PATHS = []
XML_MODEL_PATH = ""
CNF_PATH = ""


def meets_all_constraints(feature_vector):
    if csp_solver.GLOBAL_INSTANCE is not None:
        for clause in csp_solver.GLOBAL_INSTANCE.constraints['clauses']:
            if clause.is_violated_by(feature_vector):
                return False
    return True


def sort_population_by_fitness(pop):
    pop.sort(key=lambda x: x.get_fitness(0), reverse=False)


def most_common(lst):
    return max(set(lst), key=lst.count)


def generate_new_population(pop_size):
    # create new empty pop
    new_population = []

    # create children until wanted pop size is reached
    while len(new_population) <= pop_size:
        # copied_solution = copy.deepcopy(solution)
        tweaked_solution = CandidateSolution(
            csp_solver.GLOBAL_INSTANCE.generate_feature_vector())
        new_population.append(tweaked_solution)
    return new_population


def tweak(candidate):
    return candidate


def tweak_based_on_pheromones(candidate, ):
    return candidate


def adaptive_ant_mixican(generations=1, pop_size=10, best_size=1, verbose=False):
    CandidateSolution.model = feature_parser.parse(
        feature_paths=FEATURE_PATHS,
        interaction_paths=INTERACTION_PATHS,
        xml_model_path=XML_MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    cnf = CandidateSolution.model.get_cnf()
    if cnf is not None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(cnf)

    # init list for best solutions
    best_solutions = []

    # init random pheromone trails
    adaptive_evapo_rate = 1
    evapo_rate = 0.3  # 0.1

    features = CandidateSolution.model.get_features()
    feature_count = len(features)
    pheromones = {"values": {}, "rand_p": 1 / float(feature_count), "occu_counter": {}}
    for idx, f in features.items():
        if f.cnf_id != None:
            pheromones["values"][f.cnf_id] = 1 / candidate_solution.get_feature_cost(f.cnf_id)

            # occurrences counter
            pheromones["occu_counter"][f.cnf_id] = 0

    csp_solver.GLOBAL_INSTANCE.pheromones = pheromones

    # init random population
    print("Init population")
    population = []
    for i in range(pop_size):
        temp_solution = csp_solver.GLOBAL_INSTANCE.generate_feature_vector()
        if meets_all_constraints(temp_solution):
            candidate = CandidateSolution(temp_solution)
            population.append(candidate)
            print("Created candidate " + str(candidate.get_id()))

    gen_counter = 0
    best_changed = 0
    generation_info = []
    while gen_counter < generations:
        print("========== NEW GENERATION STARTED " + str(gen_counter) + str(" =========="))

        # evaporate pheromones (decrease them a bit)
        for idx in pheromones["values"]:
            val = pheromones["values"][idx]
            cost = (1 / candidate_solution.get_feature_cost(idx))
            pheromones["values"][idx] = (1 - evapo_rate) * val + evapo_rate * cost
            pheromones["occu_counter"][idx] = 0

        # assess fitness and create pheromone trail
        fitness_values = []
        fitness_sum = 0.0
        for candidate in population:
            fitness = candidate.get_fitness(0)
            fitness_values.append(fitness)
            fitness_sum += fitness
            print("Candidate id " + str(candidate.get_id()) + " has fitness: " + str(fitness))
            for idx, is_set in candidate.get_features().items():
                if is_set:
                    phero_val = pheromones["values"][idx]
                    new_value = phero_val + (1 / candidate_solution.get_feature_cost(idx) + 1 / fitness)
                    pheromones["values"][idx] = new_value
                    pheromones["occu_counter"][idx] += 1

            if len(best_solutions) < best_size:
                best_solutions.append(candidate)
            else:
                if best_solutions[-1].get_fitness(0) > fitness:
                    print("BEST CHANGED\n > new fitness", fitness)
                    best_changed = 0
                    best_solutions[-1] = candidate
                    sort_population_by_fitness(best_solutions)
        print(" > fitness average:" + str(fitness_sum / pop_size))
        print(" > best:" + str(best_solutions[0].get_fitness(0)))

        for candidate in best_solutions:
            for idx, is_set in candidate.get_features().items():
                if is_set:
                    pheromones["values"][idx] += 20 + 10 * pop_size * (1 / candidate.get_fitness(0))

        # adapt evaporation based on how often the same candidates occurred
        # different candidates -> tries different solutions based on all pheromones ("exploring")
        # same candidates -> tries to exploit best solution based its pheromones ("exploring")
        same_fitness_count = fitness_values.count(most_common(fitness_values))
        if same_fitness_count / pop_size > 0.6:
            if adaptive_evapo_rate > 1:
                adaptive_evapo_rate -= 1
        elif same_fitness_count / pop_size < 0.2:
            adaptive_evapo_rate += 1

        # average the pheromones to decrease impact of occurrences
        for idx in pheromones["values"]:
            val = pheromones["values"][idx]
            pheromones["values"][idx] = val / (max(pheromones["occu_counter"][idx], adaptive_evapo_rate))

        # if best didn't change add chance to do random changes
        # best didn't change in a while, we might be stuck in a local optimum -> break out with increasing randomness
        if best_changed > 0:
            # increase chance to ignore pheromones
            if pheromones["rand_p"] < (6 / float(feature_count)):
                pheromones["rand_p"] += 1 / float(feature_count)
        else:
            pheromones["rand_p"] = 0

        print("SAME FITNESS VALUES: ", same_fitness_count)
        print("ADAPTION RATE: ", adaptive_evapo_rate)
        print("BEST DIDN'T CHANGE SINCE: ", best_changed)
        print("BREAK OUT PROBABILITY", pheromones["rand_p"])

        # save info of generation for plotting
        generation_info.append({"best_fitness": best_solutions[0].get_fitness(0),
                                "fitness_average": fitness_sum / pop_size,
                                "adaption_rate": adaptive_evapo_rate})

        # sort by best
        sort_population_by_fitness(population)

        # tweaking based on pheromones: generate a new population
        population = generate_new_population(pop_size)

        gen_counter += 1
        best_changed += 1

    evo_result = {
        'best': {'id': best_solutions[0].get_id(), 'fitness_values': best_solutions[0].get_fitness_values()},
        'best_solutions': [],
        'population_size': pop_size,
        'generations': generations
    }
    phero_info = [(val, idx) for (idx, val) in pheromones["values"].items() if val > 0.3]
    
    for sol in best_solutions:
        print("id:" + str(sol.get_id()) + " fitness_values:" + str(sol.get_fitness_values()))
        evo_result['best_solutions'].append(sol.as_dict())

    # plot of our generation info
    plot_bar_chart_of_generation(generation_info)

    print("See log file for feature vector")

    return evo_result


def plot_bar_chart_of_generation(generation_info):
    y = [gen["fitness_average"] for gen in generation_info]
    N = len(y)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()

    best_fitness = [gen["best_fitness"] for gen in generation_info]
    rects1 = ax.bar(ind, best_fitness, width, color='r')

    fitness_average = [gen["fitness_average"] for gen in generation_info]
    rects2 = ax.bar(ind + width, fitness_average, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Overview for "+FIGURE_NAME)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple([i for i, x in enumerate(generation_info)]))

    ax.legend((rects1[0], rects2[0]), ('Best Fitness', 'Fitness Average'))


if __name__ == "__main__":
    # busy box
    # Our csp_solver might get stuck for REALLY REALLY long in backtracking steps...
    # We couldn't get this to run for more than a single generation
    # FEATURE_PATHS = ['src/project_public_2/busybox-1.198.0_feature.txt']
    # INTERACTION_PATHS = ['src/project_public_2/busybox-1.198.0_interactions.txt']
    # CNF_PATH = 'src/project_public_2/busybox-1.18.0.dimacs'
    
    ## toy box
    # set path to load DIMACS cnf formatted constraint from
    CNF_PATH = 'src/project_public_2/toybox.dimacs'
    for i in range(1,4):
        # reset number of generated candidate solutions
        # so that assigned id's are meaningful within objective optimization
        CandidateSolution.number_of_instances = 0
        
        # set path to load features values and interactions from
        # and set title of plot for this objective
        FEATURE_PATHS = ['src/project_public_2/toybox_feature'+str(i)+'.txt']
        INTERACTION_PATHS = ['src/project_public_2/toybox_interactions'+str(i)+'.txt']
        FIGURE_NAME = "Objective "+str(i)
        
        # set file_name of output log
        log_name = 'src/project_public_2/toy_box_single_log_'+str(i)+'.json'
        
        # clear previously logged content (if exists)
        json_helper.clear_json_log(log_name)
        
        # perform single state optimization
        result = adaptive_ant_mixican(
            generations=100,
            pop_size=20,
            best_size=1,
            verbose=False
        )
        
        # log results
        json_helper.extend_json_log(result, log_name)
        
        print("******************************")
        print("OPTIMIZATION "+str(i)+" DONE\n > output can be analyzed from JSON file.\n > file name:"+log_name)
    
    plt.show()
