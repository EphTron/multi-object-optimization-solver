#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

import feature_parser
import candidate_solution
from candidate_solution import CandidateSolution

def evolution(file_name, verbose, generations=50):
    features, CandidateSolution.interactions = feature_parser.parse(file_name, verbose=verbose)
    P = [candidate_solution.generate_random(features) for i in range(0,50)]
    best = None
    for gen_idx in range(0,generations):
        for p in P:
            if best is None or p.get_fitness() > best.get_fitness():
                best = p
        Q = []
        i = 0
        while i < len(P)-1:
            p1 = P[i]
            p2 = P[i+1]
            c1, c2 = candidate_solution.arbitrary_crossover(p1, p2)
            Q.append(c1)
            Q.append(c2)
            i = i+2
        P = [q for q in Q]
        if verbose:
            print("===== GENERATION ",gen_idx," =====")
            print("BEST: ", best)
            print(" > fitness", best.get_fitness())
    
    return best
    
if __name__ == "__main__":
    best = evolution('src/project_public_1/bdbc', verbose=True, generations=1000)
    print("============================= DONE! =============================")
    print("Best feature list:", best.get_feature_list())
    
