#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser
import candidate_solution
import csp_solver
import json_helper

from candidate_solution import CandidateSolution
from csp_solver import CSPSolver

import random
import utility
import copy

# used for plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

def sort_population_by_pareto_rank(pop):
    pop.sort(key=lambda x: x.pareto_rank)

def remove_same_by_fitness(candidates):
    ''' Returns list copied from candidates, where
        duplicates (comparing by fitness sum) are removed.
        remove_fitness_sums can alternatively contain
        fitness_sum values to remove from candidates. '''
    b = list(set([c.get_fitness_sum() for c in candidates]))
    temp = []
    for fitness_value in b:
        for c in candidates:
            c_val = c.get_fitness_sum()
            if abs(c_val-fitness_value) < EPSILON:
                temp.append(c)
                break
    return temp

def remove_intersecting(candidates, other_candidates):
    ''' Returns list copied from candidates, where
        intersecting (comparing by fitness sum) 
        from other_candidates are removed. '''
    b = list(set([c.get_fitness_sum() for c in other_candidates]))
    temp = []
    for c in candidates:
        c_val = c.get_fitness_sum()
        contained = False
        for fitness_value in b:
            if abs(c_val-fitness_value) < EPSILON:
                contained = True
                break
        if not contained:
            temp.append(c)
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

def generate_new_population(pop_size, check_constraints=False):
    ''' Creates list of CandidateSolutions with length equal to pop_size. 
        if check_constraints is set True given CSP constraints will be checked 
        on each feature vector generated before creating a CandidateSolution from it.
        This is genereally not necessary, as our CSP solver ensure constraints are met
        while generating feature vectors. Thus, it will not return invalid vectors.
        Flipping the check_constraints flag can be used to validate this. '''
    # create new empty pop
    new_population = []

    # create children until wanted pop size is reached
    while len(new_population) <= pop_size:
        # csp solver uses pheromones for vector generation
        vec = csp_solver.GLOBAL_INSTANCE.generate_feature_vector()
        if not check_constraints or meets_all_constraints(vec):
            new_population.append(CandidateSolution(vec))
        else:
            print "INVALID SOLUTION"
            
    return new_population
    
def pareto_burrito_acs(generations=50, pop_size=10, best_size=8, verbose=False):
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
    generation_info = {
        "previous_solutions":[],
        "pareto_front":[],
        "best_size":best_size
    }
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
                # reset sparsity (just for clean logging)
                c.sparsity = 0
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
        
        # generate a new population from pheromones
        population = generate_new_population(pop_size)

        # log current generation
        for sol in best_solutions:
            print("id:" + str(sol.get_id()) + \
                  " fitness_values:" + str(sol.get_fitness_values()) + \
                  " pareto_rank:" + str(sol.pareto_rank) + \
                  " sparsity:" + str(sol.sparsity))
        
        gen_counter += 1
        
        # store latest pareto front
        if gen_counter == generations:
            generation_info["pareto_front"] = [c for c in best_solutions if c.pareto_rank == 0]
            # if small generation count is chosen for optimization
            # algorithm might terminate before best_solution is 
            # only filled by CandidateSolution with pareo_rank = 0
            if len(generation_info["pareto_front"]) < len(best_solutions):
                generation_info["previous_solutions"].extend([c for c in best_solutions if c.pareto_rank > 0])
        # store best as previous solution
        else:
            generation_info["previous_solutions"].extend([c for c in best_solutions])
    
    result = {
        'best_solutions': [],
        'best_size': best_size,
        'population_size': pop_size,
        'generations': generations
    }
    
    phero_info = [(val, idx) for (idx, val) in pheromones["values"].items() if val > 0.3]
    #print("Pheromones", phero_info, "len", len(phero_info), "max", max([x[0] for x in phero_info]))

    for sol in best_solutions:
        result['best_solutions'].append(sol.as_dict())
    
    # process and plot collected generation info from optimization process
    finalize_generation_info(generation_info)
    plot_scatter3d_of_generations(generation_info)
    
    return result

def finalize_generation_info(generation_info):
    ''' some maintanance steps to ensure generation info contains no doubles. '''
    # make sure that previous solutions does not contain values from latest pareto front
    generation_info["previous_solutions"] = remove_intersecting(
        generation_info["previous_solutions"],
        generation_info["pareto_front"]
    )

    # remove doubles added across generations
    remove_same_by_fitness(generation_info["previous_solutions"])

def plot_scatter3d_of_generations(generation_info):
    ''' Draws a scatter plot of pareto front and all best solutions found in previous generations. '''
    if CandidateSolution.model.get_num_objectives() != 3:
        print("Failure: Can't produce 3d plot if objective count != 3")
        return
        
    # create pareto front point set
    pareto_points = np.array([c.get_fitness_values() for c in generation_info["pareto_front"]])
    
    # create previous points set
    previous_points = np.array([c.get_fitness_values() for c in generation_info["previous_solutions"]])
    
    # setup figure
    fig = plt.figure()
    ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
    pareto_scatter = ax.scatter(pareto_points[:,0],pareto_points[:,1],pareto_points[:,2],color='red')
    prev_gen_scatter = ax.scatter(previous_points[:,0],previous_points[:,1],previous_points[:,2])
    ax.set_xlabel("Objective 1 values")
    ax.set_ylabel("Objective 2 values")
    ax.set_zlabel("Objective 3 values")
    ax.set_title("Pareto Front and best Candidates of previous Generations")
    pareto_label = 'Pareto Front\n(if "Number of Candidates at rank 0" > '+str(generation_info["best_size"])+': shows '+str(generation_info["best_size"])+' elements maximizing sparsity)'
    prev_gen_label = 'Best Candidates of previous Generations'
    ax.legend((pareto_scatter, prev_gen_scatter), (pareto_label, prev_gen_label))
    # invert axis because we optimize towards lowest values
    # and pareto front should be outmost points
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
        
if __name__ == "__main__":
    # toy box
    FEATURE_PATHS.append('src/project_public_2/toybox_feature1.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature2.txt')
    FEATURE_PATHS.append('src/project_public_2/toybox_feature3.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions1.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions2.txt')
    INTERACTION_PATHS.append('src/project_public_2/toybox_interactions3.txt')
    CNF_PATH = 'src/project_public_2/toybox.dimacs'

    result = pareto_burrito_acs(
        generations=50,
        pop_size=10,
        best_size=8,
        verbose=False
    )
    
    log_name = 'src/project_public_2/toybox_multi_log.json'
    json_helper.clear_json_log(log_name)
    json_helper.extend_json_log(result, log_name)
    
    print("******************************")
    print("OPTIMIZATION DONE\n > output can be analyzed from JSON file.\n > file name:" + log_name)
    
    plt.show()