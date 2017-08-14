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

EPSILON = 0.0000000001
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
                # copied_solution = copy.deepcopy(solution)
                tweaked_solution = CandidateSolution(
                    csp_solver.GLOBAL_INSTANCE.generate_feature_vector())  # tweak(copied_solution)
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
    adaptive_evapo_rate = 1
    evapo_rate = 0.3  # 0.1
    learn_rate = 0.5  # 0.5

    pheromones = {"values": {}, "rand_p": 0.0, "occu_counter": {}}
    features = CandidateSolution.model.get_features()
    for f_name, f in features.items():
        if f.cnf_id != None:
            cost = 0
            for objective_id in range(0, CandidateSolution.model.get_num_objectives()):
                cost += (1 / candidate_solution.get_feature_cost(f.cnf_id, objective_id))
            pheromones["values"][f.cnf_id] = cost

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
            pheromones["occu_counter"][idx] = 0
            # pheromones["values"][idx] = (1 - evapo_rate) * val + evapo_rate * cost # base_value

        # extend population with best from previous generation
        population.extend(best_solutions)
        # generate pareto properties and sort population accordingly
        calc_pareto_ranks(population)
        sort_population_by_pareto_rank(population)
        '''
        for candidate in population:
            print("================")
            print(" > pareto_rank:"+str(candidate.pareto_rank))
            print(" > values:"+str(candidate.get_fitness_values()))
        '''
        # append to best solutions from pareto ranked
        best_solutions = []
        for c in population:
            if c.pareto_rank > 0 and len(best_solutions) >= best_size:
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
        
        # TODO: handle this
        if len(best_solutions) < best_size:
            print("best size not reached\n > size is"+str(len(best_solutions)))
        #while len(best_solutions) < best_size:
        #    best_solutions.append(best_solutions[0])
        
        
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
                    # pheromones["occu_counter"][idx] += 1
                    
        ##############################
        
        # select best solutions for breeding
        breeding_q = population[:selection_size]

        # breeding: first copy, then tweak or crossover to generate a new population
        population = breed(breeding_q, pop_size)

        gen_counter += 1
        # print("Pheromones", pheromones["values"])
        # print("Occurrences", pheromones["occu_counter"])
        
        for sol in best_solutions:
            print("id:" + str(sol.get_id()) + " fitness_values:" + str(sol.get_fitness_values()) + " pareto_rank:" + str(sol.pareto_rank))
    

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


def test_csp_solver(file_name, verbose):
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

    for i in range(0, 250):
        print("============== CANDIDATE " + str(i) + " =============")
        vec = {
            cnf_id: random.randint(0, 1) > 0 for cnf_id in cnf['cnf_id_to_f_name'].keys()
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

    # busy box
    # FEATURE_PATHS.append('src/project_public_2/busybox-1.198.0_feature.txt')
    # INTERACTION_PATHS.append( 'src/project_public_2/busybox-1.198.0_interactions.txt')
    # CNF_PATH = 'src/project_public_2/busybox-1.18.0.dimacs'

    # toy box
    FEATURE_PATHS.append('src/project_public_2/toybox_feature1.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature2.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature3.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions1.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions2.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions3.txt')
    CNF_PATH = 'src/project_public_2/toybox.dimacs'

    result = simple_evolution_template(
        generations=200,
        pop_size=20,
        selection_size=5,
        best_size=5,
        verbose=False
    )
    
    json_helper.clear_json_log('src/project_public_2/toy_box_pheromones.json')
    json_helper.extend_json_log(result, 'src/project_public_2/toy_box_pheromones.json')