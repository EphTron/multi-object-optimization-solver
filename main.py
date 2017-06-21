#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21.06.17 14:30
@author: ephtron
"""


def create_dict_from_txt(path):
    """
    Creates a feature or interaction dict from the given txt file
    :param path: string
    :return: dict
    """
    # read in feature and interaction files
    with open(path, "r") as file:
        features = file.read().replace(" ", "").split('\n')
    print("Collected features: ", features)

    # create dict out of the given features
    pairs = [feature.split(":") for feature in features if feature is not ""]
    feature_dict = dict((k.strip(), float(v.strip())) for k, v in pairs)
    print("Feature dictonary", feature_dict)


if __name__ == "__main__":
    features = create_dict_from_txt("src/project_public_1/bdbc_feature.txt")
    interactions = create_dict_from_txt("src/project_public_1/bdbc_interactions.txt")
