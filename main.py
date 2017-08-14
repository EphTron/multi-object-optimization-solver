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
    pop.sort(key=lambda x: x.get_fitness(), reverse=False)


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
    evapo_rate = 0.1  # 0.1
    learn_rate = 0.5  # 0.5
    pheromones = {"values": {}, "rand_p": 1.0, "occu_counter": {}}
    features = CandidateSolution.model.get_features()
    for idx, f in features.items():
        if f.cnf_id != None:
            pheromones["values"][f.cnf_id] = 1 / candidate_solution.get_feature_cost(f.cnf_id)

            # occurrences counter
            # pheromones["occu_counter"][f.cnf_id] = 0

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
    while gen_counter < generations:
        pheromones["rand_p"] = max(min(1 - (gen_counter / float(generations)), 0.8), 0.1)
        print("========== NEW GENERATION STARTED " + str(gen_counter) + str(" =========="))
        fitness_sum = 0.0

        # evapo pheromones (decrease them a bit)
        for idx in pheromones["values"]:
            val = pheromones["values"][idx]
            cost = (1 / candidate_solution.get_feature_cost(idx))
            pheromones["values"][idx] = (1 - evapo_rate) * val + evapo_rate * cost

            # pheromones["occu_counter"][f.cnf_id] = 0
            # pheromones["values"][idx] = (1 - evapo_rate) * val + evapo_rate * cost # base_value

        # assess fitness and create pheromone trail        
        for candidate in population:
            # print(len(population))
            fitness = candidate.get_fitness()
            fitness_sum += fitness
            print("Candidate id " + str(candidate.get_id()) + " has fitness: " + str(fitness))
            for idx, is_set in candidate.get_features().items():
                if is_set:
                    pheromones["values"][idx] += 1 / candidate_solution.get_feature_cost(idx) + 1 / fitness

                    # other pheromone settings approach
                    # pheromones["occu_counter"][f.cnf_id] += 1
                    # desirability = 1 / candidate_solution.get_feature_cost(idx) + 1 / fitness
                    # pheromones["values"][idx] = (1 - learn_rate) * val + learn_rate * desirability

            if len(best_solutions) < best_size:
                best_solutions.append(candidate)
            else:
                if best_solutions[-1].get_fitness() > fitness:
                    print("BEST CHANGED\n > new fitness", fitness)
                    best_solutions[-1] = candidate
                    sort_population_by_fitness(best_solutions)
        print(" > fitness sum:" + str(fitness_sum) + "\n > fitness average:" + str(fitness_sum / pop_size))
        print(" > best:" + str(best_solutions[0].get_fitness()))

        for candidate in best_solutions:
            for idx, is_set in candidate.get_features().items():
                if is_set:
                    # val = pheromones["values"][idx]
                    # print idx, pheromones["values"][idx]
                    # fitness =
                    # pheromones["values"][idx] += 0.5 * (1 / candidate.get_fitness())
                    # pheromones["values"][idx] += learn_rate * (1 / candidate.get_fitness())
                    # pheromones["occu_counter"][idx] += 1
                    pass

        # average the pheromones to decrease impact of occurrences
        # for idx in pheromones["values"]:
        #    val = pheromones["values"][idx]
        #    pheromones["values"][idx] = val / max(pheromones["occu_counter"][idx], 1)

        # sort by best
        sort_population_by_fitness(population)

        # select best solutions for breeding
        breeding_q = population[:selection_size]

        # breeding: first copy, then tweak or crossover to generate a new population
        population = breed(breeding_q, pop_size)

        gen_counter += 1
        print(pheromones["values"])

    result = {
        'best': {'id': best_solutions[0].get_id(), 'fitness': best_solutions[0].get_fitness()},
        'best_solutions': [],
        'population_size': pop_size,
        'generations': generations,
        'selection_size': selection_size
    }

    for sol in best_solutions:
        print("id:" + str(sol.get_id()) + " fitness:" + str(sol.get_fitness()))
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
    if cnf != None:
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
    FEATURE_PATHS.append('src/project_public_2/toybox_feature1.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature2.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature3.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions1.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions2.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions3.txt')
    CNF_PATH = 'src/project_public_2/toybox.dimacs'
    
    result = simple_evolution_template(
        generations=20,
        pop_size=25,
        selection_size=5,
        best_size=1,
        verbose=False
    )
    
    json_helper.clear_json_log('src/project_public_2/toy_box_pheromones.json')
    json_helper.extend_json_log(result, 'src/project_public_2/toy_box_pheromones.json')