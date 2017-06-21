#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""

def assess_fitness(feature_list):
    feature_fitness = 0
    for feature in feature_list:
        feature_fitness += feature.value

    interaction_fitness = 0
    # TODO: add

    return



import feature_parser


if __name__ == "__main__":
    features, interactions = feature_parser.parse('src/project_public_1/bdbc')
    print '#############FEATURES##############'
    for key in features:
        print '==========\n', features[key]
    print '#############INTERACTIONS##############'
    for i in interactions:
        print '==========\n', i
