#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser
import candidate_solution
from candidate_solution import CandidateSolution
import csp_solver
from csp_solver import CSPSolver

import random
import utility
import copy
import json
import os
import os.path
import datetime

# used for plotting
import numpy
import matplotlib.pyplot as plt

FEATURE_PATH = ""
INTERACTION_PATH = ""
MODEL_PATH = ""
CNF_PATH = ""

def write_json_to_file(json_dict, file_name):
    ''' (over)writes contents of json_dict into file referenced by file_name. '''
    with open(file_name, 'w') as file:
        json.dump(json_dict, file, sort_keys=True, indent=4, separators=(',', ': '))
        file.close()

def extend_json_log(json_dict, file_name):
    ''' appends contents of json_dict to logging structure 
        in JSON file referenced by file_name. '''
    time_stamp = str(datetime.datetime.now())
    full_json = None 
    if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
        full_json = {time_stamp:json_dict}
    else:
        with open(file_name, 'r') as file:
            full_json = json.load(file)
            file.close()
        full_json[time_stamp] = json_dict
    if 'best' in json_dict:
        if 'best' in full_json:
            if json_dict['best']['fitness'] < full_json['best']['fitness']:
                full_json['best'] = json_dict['best']
                full_json['best']['time_stamp'] = time_stamp
        else:
            full_json['best'] = json_dict['best']
            full_json['best']['time_stamp'] = time_stamp
    write_json_to_file(full_json, file_name)
    
def clear_json_log(file_name):
    write_json_to_file({}, file_name)

def brute_force(file_name, verbose):
    features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        file_name,
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    if CandidateSolution.cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    best = None
    max_count = 2 ** len(features) - 1
    last_str = '{0:b}'.format(max_count)
    c_feature_names = [f_name for f_name in features.keys()]
    for i in range(0, max_count):
        if verbose and i % 100 == 0:
            print(i, " of ", max_count)
        str = '{0:b}'.format(i)
        while len(str) != len(last_str):
            str = '0' + str
        c_features = {f_name: None for f_name in features.keys()}
        for digit in range(0, len(str)):
            if str[digit] == '1':
                c_features[c_feature_names[digit]] = features[c_feature_names[digit]]
        c = CandidateSolution(c_features)
        if c.is_valid() and (best is None or best.get_fitness() < c.get_fitness()):
            best = c
    return best


def naive_evolution(file_name, verbose, generations=50, population_size=10):
    features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        file_name,
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    if CandidateSolution.cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    P = [candidate_solution.generate_random(features) for i in range(0, population_size)]
    best = None
    best_gen = 0

    fig = plt.figure()
    ax = plt.subplot(111)
    for gen_idx in range(0, generations):
        utility.plot_generation_boxplot(ax, P, gen_idx)
        for p in P:
            if best is None or p.get_fitness() > best.get_fitness():
                best = p
                best_gen = gen_idx
        Q = []
        i = 0

        while i < len(P) - 1:
            p1 = P[i]
            p2 = P[i + 1]
            c1, c2 = candidate_solution.arbitrary_crossover(p1, p2)
            Q.append(c1)
            Q.append(c2)
            i = i + 2
        P = [q for q in Q]
        if verbose:
            print("===== GENERATION ", gen_idx, " =====")
            print("BEST: ", best)
            print(" > fitness", best.get_fitness())

    return best, best_gen


def partially_random_evolution(file_name, verbose, generations=50, population_size=10):
    features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        file_name,
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    if CandidateSolution.cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    for i in range(0, 100):
        print("============== DEINE MUTTER =============")
        print(csp_solver.GLOBAL_INSTANCE.generate_feature_vector())
        print("============== ENDE =============")
    return
    P = [candidate_solution.generate_random(features) for i in range(0, population_size)]
    best = None
    best_gen = 0

    fig = plt.figure()
    ax = plt.subplot(111)
    for gen_idx in range(0, generations):
        utility.plot_generation_boxplot(ax, P, gen_idx)
        for p in P:
            if best is None or p.get_fitness() > best.get_fitness():
                best = p
                best_gen = gen_idx
        Q = []
        i = 0
        while i < len(P) / 2 - 1:
            p1 = P[i]
            p2 = P[i + 1]
            c1, c2 = candidate_solution.arbitrary_crossover(p1, p2)
            Q.append(c1)
            Q.append(c2)
            i = i + 2
        P = [q for q in Q]
        while len(P) < population_size - 1:
            P.append(candidate_solution.generate_random(features))
        random.shuffle(P)
        if verbose:
            print("===== GENERATION ", gen_idx, " =====")
            print("BEST: ", best)
            print(" > fitness", best.get_fitness())

    return best, best_gen


def meets_all_constraints(feature_vector):
    if csp_solver.GLOBAL_INSTANCE != None:
        for clause in csp_solver.GLOBAL_INSTANCE.constraints['clauses']:
            if clause.is_violated_by(feature_vector):
                return False
    return True


def sort_population_by_fitness(pop):
    pop.sort(key=lambda x: x.get_fitness(), reverse=False)


def breed(solutions_for_breeding, pop_size):
    # create new empty pop
    new_population = []

    # create children until wanted pop size is reached
    while len(new_population) <= pop_size:
        for solution in solutions_for_breeding:
            if len(new_population) <= pop_size:
                copied_solution = copy.deepcopy(solution)
                tweaked_solution = tweak(copied_solution)
                new_population.append(tweaked_solution)
    return new_population


def tweak(candidate):
    return candidate


def simple_evolution_template(generations=1, pop_size=10, selection_size=5, verbose=False):
    # init parser, features, interactions
    CandidateSolution.features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        "obsolete",
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    # CandidateSolution.features,
    # print("ALL ", features)

    if CandidateSolution.cnf is not None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    # init list for best solutions
    best_solutions = []

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
    while gen_counter < generations:

        # assess fitness and create pheromone trail
        for candidate in population:
            print len(population)
            fitness = candidate.get_fitness()
            print("Candidate id " + str(candidate.get_id()) + " has fitness: " + str(fitness))
            if len(best_solutions) <= 5:
                best_solutions.append(candidate)
            else:
                if best_solutions[-1].get_fitness() > fitness:
                    best_solutions[-1] = candidate
                    sort_population_by_fitness(best_solutions)

        # sort by best
        sort_population_by_fitness(population)

        # select best solutions for breeding
        breeding_q = population[:selection_size]

        # breeding: first copy, then tweak or crossover to generate a new population
        population = breed(breeding_q, pop_size)

        gen_counter += 1
    
    result = {
        'best':{'id':best_solutions[0].get_id(), 'fitness':best_solutions[0].get_fitness()},
        'best_solutions':[], 
        'population_size':pop_size, 
        'generations':generations, 
        'selection_size':selection_size
    }
    
    for sol in best_solutions:
        print("id:"+str(sol.get_id())+" fitness:"+str(sol.get_fitness()))
        result['best_solutions'].append(sol.as_dict())
    
    return result


def test_csp_solver(file_name, verbose):
    features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        file_name,
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    if CandidateSolution.cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    for i in range(0, 250):
        print("============== CANDIDATE " + str(i) + " =============")

        vec = csp_solver.GLOBAL_INSTANCE.generate_feature_vector()
        print(vec)
        if meets_all_constraints(vec):
            print(" > Meets all constraints")
        else:
            print(" > Does not meet all constraints")
    return


def test_fix_vector(file_name, verbose):
    features, CandidateSolution.interactions, CandidateSolution.cnf = feature_parser.parse(
        file_name,
        feature_path=FEATURE_PATH,
        interaction_path=INTERACTION_PATH,
        model_path=MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    if CandidateSolution.cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(CandidateSolution.cnf)

    for i in range(0, 250):
        print("============== CANDIDATE " + str(i) + " =============")
        vec = {
            cnf_id: random.randint(0, 1) > 0 for cnf_id in CandidateSolution.cnf['cnf_id_to_f_name'].keys()
        }
        print(vec)
        vec = csp_solver.GLOBAL_INSTANCE.fix_feature_vector(vec)
        print(vec)
        if vec is not None and meets_all_constraints(vec):
            print(" > Meets all constraints")
        elif vec is None:
            print(" > Candidate is None.")            
        else:
            print(" > Does not meet all constraints")
    return


if __name__ == "__main__":
    FEATURE_PATH = 'src/project_public_2/toybox_feature1.txt'
    INTERACTION_PATH = 'src/project_public_2/toybox_interactions1.txt'
    CNF_PATH = 'src/project_public_2/toybox.dimacs'
    
    # clear_json_log('src/project_public_2/toy_box_res.json')
    extend_json_log(simple_evolution_template(1, 10000, 5), 'src/project_public_2/toy_box_test.json')
    
    """
    FEATURE_PATH = 'src/project_public_2/toybox_feature1.txt'
    INTERACTION_PATH = 'src/project_public_2/toybox_interactions1.txt'
    CNF_PATH = 'src/project_public_2/toybox.dimacs'
    test_fix_vector('src/project_public_1/toybox', verbose=True)
    
    FEATURE_PATH = 'src/project_public_2/busybox-1.198.0_feature.txt'
    INTERACTION_PATH = 'src/project_public_2/busybox-1.198.0_interactions.txt'
    CNF_PATH = 'src/project_public_2/busybox-1.18.0.dimacs'
    test_csp_solver('src/project_public_1/busybox', verbose=True)
    """
