#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""
import feature_parser

if __name__ == "__main__":
    features, interactions = feature_parser.parse('src/project_public_1/bdbc', verbose=False)
