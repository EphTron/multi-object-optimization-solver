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
import json_helper

import random
import utility
import copy

# used for plotting
import numpy
import matplotlib.pyplot as plt

EPSILON  =  0.0000000001
INFINITY = 9999999999
FEATURE_PATHS = []
INTERACTION_PATHS = []
XML_MODEL_PATH = ""
CNF_PATH = ""

def meets_all_constraints(feature_vector):
    if csp_solver.GLOBAL_INSTANCE != None:
        for clause in csp_solver.GLOBAL_INSTANCE.constraints['clauses']:
            if clause.is_violated_by(feature_vector):
                return False
    return True


def sort_population_by_fitness(pop):
    pop.sort(key=lambda x: x.get_fitness(0), reverse=False)

def sort_population_by_pareto_rank(pop):
    pop.sort(key=lambda x: x.pareto_rank)

def most_common(lst):
    return max(set(lst), key=lst.count)

def remove_same_by_fitness(candidates):
    b = list(set([c.get_fitness_sum() for c in candidates]))
    temp = []
    for fitness_value in b:
        for c in candidates:
            c_val = c.get_fitness_sum()
            if abs(c_val-fitness_value) < EPSILON:
                temp.append(c)
                break
    return temp

def calc_sparsity(candidates):
    ''' Calculates sparsity for all candidates. 
        Note all elements should be of same pareto rank. '''
    for c in candidates:
        c.sparsity = 0
    sorted_candidates = [c for c in candidates]
    for i in range(0, CandidateSolution.model.get_num_objectives()):
        sorted_candidates.sort(key=lambda x: x.get_fitness(i), reverse=False)
        sorted_candidates[0].sparsity = INFINITY 
        sorted_candidates[-1].sparsity = INFINITY
        v_range = sorted_candidates[-1].get_fitness(i) - sorted_candidates[0].get_fitness(i)
        for j in range(1, len(sorted_candidates)-1):
            current = sorted_candidates[j]
            prev = sorted_candidates[j-1].get_fitness(i)
            next = sorted_candidates[j+1].get_fitness(i)
            current.sparsity += (next - prev) / float(v_range)

def calc_pareto_front(candidates):
    ''' given a list of CandidateSolution this function 
        computes the Pareto Front. '''
    front = []
    for c in candidates:
        kill_list = []
        front.append(c)
        remove_c = False
        for f_c in front:
            if c == f_c:
                continue
            if c.dominates(f_c):
                kill_list.append(f_c)
            elif f_c.dominates(c):
                remove_c = True
                break
        if remove_c:
            front.remove(c)
        else:
            while len(kill_list) > 0:
                front.remove(kill_list[-1])
                kill_list.pop()
    return front
    
def calc_pareto_ranks(candidates, i=0):
    ''' recursively computes pareto ranks for all candidates. '''
    if len(candidates) == 0:
        return
        
    front = calc_pareto_front(candidates)
    for c in front:
        c.pareto_rank = i
    
    calc_pareto_ranks([c for c in candidates if c not in front], i+1)

def breed(solutions_for_breeding, pop_size):
    # create new empty pop
    new_population = []

    # create children until wanted pop size is reached
    while len(new_population) <= pop_size:
        for solution in solutions_for_breeding:
            if len(new_population) <= pop_size:
                # csp solver uses pheromones for vector generation
                tweaked_solution = CandidateSolution(
                    csp_solver.GLOBAL_INSTANCE.generate_feature_vector()
                )
                new_population.append(tweaked_solution)
    return new_population

def tweak(candidate):
    return candidate

def tweak_based_on_pheromones(candidate, ):
    return candidate

def simple_evolution_template(generations=1, pop_size=10, selection_size=5, best_size=1, verbose=False):
    CandidateSolution.model = feature_parser.parse(
        feature_paths=FEATURE_PATHS,
        interaction_paths=INTERACTION_PATHS,
        xml_model_path=XML_MODEL_PATH,
        cnf_path=CNF_PATH,
        verbose=verbose
    )
    cnf = CandidateSolution.model.get_cnf()
    if cnf != None:
        csp_solver.GLOBAL_INSTANCE = CSPSolver(cnf)
    
    # init list for best solutions
    best_solutions = []

    # init random pheromone trails
    evapo_rate = 0.3  # 0.3
    learn_rate = 0.5  # 0.5
    
    # initialize pheromones with summed feature cost for all objectives
    # rand_p only used for single state optimization
    pheromones = {"values": {}, "rand_p": 0.0, "occu_counter": {}}
    features = CandidateSolution.model.get_features()
    for f_name, f in features.items():
        if f.cnf_id != None:
            cost = 0
            for objective_id in range(0, CandidateSolution.model.get_num_objectives()):
                cost += (1 / candidate_solution.get_feature_cost(f.cnf_id, objective_id))
            pheromones["values"][f.cnf_id] = cost

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
    feature_count = len(features)
    while gen_counter < generations:
        print("========== NEW GENERATION STARTED " + str(gen_counter) + str(" =========="))
        fitness_sum = 0.0

        # evapo pheromones (decrease them a bit)
        for idx in pheromones["values"]:
            val = pheromones["values"][idx]
            cost = 0
            for objective_id in range(0, CandidateSolution.model.get_num_objectives()):
                cost += (1 / candidate_solution.get_feature_cost(idx, objective_id))
            pheromones["values"][idx] = (1 - evapo_rate) * val + evapo_rate * cost
            
        # extend population with best from previous generation
        population.extend(best_solutions)
        
        # generate pareto properties and sort population accordingly
        calc_pareto_ranks(population)
        sort_population_by_pareto_rank(population)
        
        # append to best solutions from pareto ranked
        best_solutions = []
        rank_0 = remove_same_by_fitness([c for c in population if c.pareto_rank == 0])
        if len(rank_0) >= best_size:
            calc_sparsity(rank_0)
            rank_0.sort(key=lambda x: x.sparsity, reverse=True)
            best_solutions = rank_0[:best_size]
        else:
            rank_other = [c for c in best_solutions if c.pareto_rank != 0]
            for c in population:
                if len(best_solutions) >= best_size:
                    break
                same = False
                for bc in best_solutions:
                    c_val = sum(c.get_fitness_values())
                    bc_val = sum(bc.get_fitness_values())
                    if abs(c_val-bc_val) < EPSILON:
                        same = True
                        break
                if not same:
                    best_solutions.append(c)
        
        if len(best_solutions) < best_size:
            print("best size not reached\n > size is"+str(len(best_solutions)))
        
        # get min and max fitness values 
        # for all objectives of best candidates
        min_v = []
        max_v = []
        vals = []
        for i in range(0, CandidateSolution.model.get_num_objectives()):
            min_v.append(min([c.get_fitness(i) for c in best_solutions]))
            max_v.append(max([c.get_fitness(i) for c in best_solutions]))
            
        
        # pheromones for best
        for candidate in best_solutions:
            # map fitness value to range [1,100] based on best candidates [min, max]
            # for all objectives
            mapped_fitness_vals = []
            for i in range(0, CandidateSolution.model.get_num_objectives()):
                mapped_fitness_vals.append(
                    candidate_solution.map_to_range(
                        candidate.get_fitness(i), 
                        min_v[i], max_v[i], 1, 100
                    )
                )
            for idx, is_set in candidate.get_features().items():
                if is_set:
                    pheromones["values"][idx] += 20 + 10 * pop_size * 1/sum(mapped_fitness_vals)
        
        # select best solutions for breeding
        breeding_q = population[:selection_size]

        # breeding: first copy, then tweak or crossover to generate a new population
        population = breed(breeding_q, pop_size)

        gen_counter += 1
        
        for sol in best_solutions:
            print("id:" + str(sol.get_id()) + \
                  " fitness_values:" + str(sol.get_fitness_values()) + \
                  " pareto_rank:" + str(sol.pareto_rank) + \
                  " sparsity:" + str(sol.sparsity))
    

    result = {
        'best': {'id': best_solutions[0].get_id(), 'fitness_values': best_solutions[0].get_fitness_values()},
        'best_solutions': [],
        'population_size': pop_size,
        'generations': generations,
        'selection_size': selection_size
    }
    phero_info = [(val, idx) for (idx, val) in pheromones["values"].items() if val > 0.3]
    print("Pheromones", phero_info, "len", len(phero_info), "max", max([x[0] for x in phero_info]))

    for sol in best_solutions:
    #    print("id:" + str(sol.get_id()) + " fitness_values:" + str(sol.get_fitness_values()))
        result['best_solutions'].append(sol.as_dict())

    return result

if __name__ == "__main__":
    # toy box
    FEATURE_PATHS.append('src/project_public_2/toybox_feature1.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature2.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature3.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions1.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions2.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions3.txt')
    CNF_PATH = 'src/project_public_2/toybox.dimacs'

    result = simple_evolution_template(
        generations=100,
        pop_size=10,
        selection_size=5,
        best_size=8,
        verbose=False
    )
    
    json_helper.clear_json_log('src/project_public_2/toy_box_pheromones.json')
    json_helper.extend_json_log(result, 'src/project_public_2/toy_box_pheromones.json')