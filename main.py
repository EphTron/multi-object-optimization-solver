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

# used for plotting
import numpy
import matplotlib.pyplot as plt

FEATURE_PATH = ""
INTERACTION_PATH = ""
MODEL_PATH = ""
CNF_PATH = ""

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
    max_count = 2**len(features)-1
    last_str = '{0:b}'.format(max_count)
    c_feature_names = [f_name for f_name in features.keys()]
    for i in range(0, max_count):
        if verbose and i % 100 == 0:
            print(i, " of ", max_count)
        str = '{0:b}'.format(i)
        while len(str) != len(last_str):
            str = '0'+str
        c_features = {f_name:None for f_name in features.keys()}
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
        
    P = [candidate_solution.generate_random(features) for i in range(0,population_size)]
    best = None
    best_gen = 0
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for gen_idx in range(0,generations):
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
    
    for i in range(0,100):
        print("============== DEINE MUTTER =============")
        print(csp_solver.GLOBAL_INSTANCE.generate_feature_vector())
        print("============== ENDE =============")
    return
    P = [candidate_solution.generate_random(features) for i in range(0,population_size)]
    best = None
    best_gen = 0
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for gen_idx in range(0,generations):
        utility.plot_generation_boxplot(ax, P, gen_idx)
        for p in P:
            if best is None or p.get_fitness() > best.get_fitness():
                best = p
                best_gen = gen_idx
        Q = []
        i = 0
        while i < len(P)/2-1:
            p1 = P[i]
            p2 = P[i+1]
            c1, c2 = candidate_solution.arbitrary_crossover(p1, p2)
            Q.append(c1)
            Q.append(c2)
            i = i+2
        P = [q for q in Q]
        while len(P) < population_size-1:
            P.append(candidate_solution.generate_random(features))
        random.shuffle(P)
        if verbose:
            print("===== GENERATION ",gen_idx," =====")
            print("BEST: ", best)
            print(" > fitness", best.get_fitness())
    
    return best, best_gen

if __name__ == "__main__":
    FEATURE_PATH = 'src/project_public_2/toybox_feature1.txt'
    INTERACTION_PATH = 'src/project_public_2/toybox_interactions1.txt'
    CNF_PATH = 'src/project_public_2/toybox.dimacs'
    best, best_gen = partially_random_evolution('src/project_public_1/toybox', verbose=True, generations=100)
    #best, best_gen = partially_random_evolution('src/project_public_1/bdbc', verbose=True, generations=100)
    print("============================= DONE! =============================")
    print(" > feature list:", [f.name for f in best.get_feature_list()])
    print(" > fitness:", best.get_fitness())
    print(" > generation found:", best_gen)
    # print("============================= DONE! =============================")
    # print("Best feature list:", best.get_feature_list())
    plt.show()
