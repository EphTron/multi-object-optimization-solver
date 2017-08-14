from xml.dom import minidom
from feature import Feature
from constraint_clause import ConstraintClause
from feature_interaction import FeatureInteraction
from feature_model import FeatureModel
import os.path


def parse(feature_paths, interaction_paths, xml_model_path="", cnf_path="", verbose=False):
    if len(feature_paths) < 1:
        raise ValueError("Failure: parse(...): Didn't provide at least one filepath to a feature description.")
    elif len(interaction_paths) < 1:
        raise ValueError("Failure: parse(...): Didn't provide provide at least one filepath to a feature interaction description.")
    
    fp = FeatureParser()
    
    # create initial model
    # adding base features and constraints
    model = FeatureModel()
    model.set_features(fp.parse_features(feature_paths[0], xml_model_path))
    model.set_cnf(fp.parse_cnf(cnf_path, model.get_features()))
    
    # add all interactions
    for path in interaction_paths:
        model.add_interaction_set(fp.parse_interactions(path))
    
    # add additional feature values if any
    if len(feature_paths) > 1:
        for i in range(1,len(feature_paths)):
            fp.extend_feature_values(model.get_features(), feature_paths[i])
            
    if verbose:
        print('#############FEATURES##############')
        features = model.get_features()
        for name in features:
            print(features[name])
        for i in range(0, model.get_num_interaction_sets()):
            print('#############INTERACTIONS '+str(i)+'##############')
            for interaction in model.get_interaction_set(i):
                print(interaction)
        print('#############CNF CLAUSES#############')
        cnf = model.get_cnf()
        print(' > p_line:', cnf['p_line'])
        print(' > clauses:')
        for c in cnf['clauses']:
            print(c)
    return model

class FeatureParser(object):
    def __init__(self):
        pass
    
    def parse_features(self, feature_path, xml_model_path=""):
        ''' Parses only features from given feature_path. 
            Can extend feature description with info stored in xml_model_path. '''
        features = {
            name: Feature(name, value)
            for name, value in self._create_dict_from_txt(feature_path).items()
        }
        
        # check to see if xml model exists and extend features if found
        if len(xml_model_path) > 0 and os.path.isfile(xml_model_path):
            self._parse_xml_model(features, xml_model_path)
        
        return features

    def parse_interactions(self, interaction_path):
        ''' parses only interactions from given interaction_path. '''
        return [
            FeatureInteraction(key.split('#'), value)
            for key, value in self._create_dict_from_txt(interaction_path).items()
        ]
    
    def parse_cnf(self, cnf_path, features):
        ''' Parses constraints in cnf format from DIMACS file referenced by cnf_path.
            features needed for name lookup and appending cnf_id to each feature. '''
        # init constraint list
        cnf = {'p_line': None, 'clauses': [], 'cnf_id_to_f_name': {}}

        # check to see if cnf file exists and extend features if found
        if len(cnf_path) > 0 and os.path.isfile(cnf_path):
            cnf = self._parse_cnf(cnf_path, features)
        
        return cnf
    
    def extend_feature_values(self, features, feature_path):
        ''' extends given feature dict by values read from file. '''
        for name, value in self._create_dict_from_txt(feature_path).items():
            if name in features:
                features[name].add_value(value)
                
    ##################################################################
    ######################## HELPER FUNCTIONS ########################
    ##################################################################
    
    def _create_dict_from_txt(self, path):
        """
        Creates a feature or interaction dict from the given txt file
        :param path: string
        :return: dict
        """
        # read in feature and interaction files
        with open(path, "r") as file:
            features = file.read().replace(" ", "").split('\n')
        # create dict out of the given features
        pairs = [feature.split(":") for feature in features if feature is not ""]
        feature_dict = dict((k.strip(), float(v.strip())) for k, v in pairs)
        return feature_dict

    def _parse_xml_model(self, features, f_name):
        ''' HELPER FUNCTION for parse.
            parses an xml model file with given f_name
            and extends given features with data retrieved. '''
        # parse xml doc to get xml_feature list
        xml_doc = minidom.parse(f_name)
        xml_feature_list = xml_doc.getElementsByTagName('configurationOption')

        # extract info for all retrieved model features
        feature_excludes = {}
        for xml_feature in xml_feature_list:
            # extract name of feature and continue only if name is set
            f_name = self._get_text_of_field(xml_feature, 'name')
            if len(f_name) == 0:
                continue
            # create Feature and set class properties
            f = features[f_name]
            f.output_string = self._get_text_of_field(xml_feature, 'outputString')
            f.default_value = self._get_text_of_field(xml_feature, 'defaultValue')
            f.optional = self._get_text_of_field(xml_feature, 'optional')
            # store string list of exclude features for later
            xml_exclude_options = xml_feature.getElementsByTagName('excludedOptions')[0]
            if len(xml_exclude_options.childNodes) > 0:
                feature_excludes[f_name] = self._get_text_list_of_fields(xml_exclude_options, 'options')
            else:
                feature_excludes[f_name] = []

        # compile list of exclude_features for each Feature
        # given string dictionary extracted from xml
        for key in features:
            if key not in feature_excludes or len(feature_excludes[key]) == 0:
                pass
            else:
                f = features[key]
                f.exclude_features = [features[f_name] for f_name in feature_excludes[key]]

    def _parse_cnf(self, file_name, features):
        ''' HELPER function for parse_cnf(...).
            Parses a DIMACS CNF file to extend features with cnf_id
            and returns the list of clauses extracted from file. '''
        cnf = {'p_line': None, 'clauses': [], 'cnf_id_to_f_name': {}}
        # read in feature and interaction files
        with open(file_name, "r") as file:
            # remove '$ ' symbols preceeding feature names
            lines = file.read().replace("$ ", " ").split("\n")
        cnf['cnf_id_to_f_name'] = {}
        clauses = []
        p_line = {'cnf': '', 'nbvar': 0, 'nbclauses': 0}
        current_clause = []
        for line in lines:
            words = line.split(" ")
            if len(words) == 0:
                continue
            if words[0] == "c":
                if len(words) != 3:
                    continue
                if words[2] not in features:
                    continue
                features[words[2]].cnf_id = int(words[1])
                cnf['cnf_id_to_f_name'][int(words[1])] = words[2]
            elif words[0] == "p":
                if len(words) != 4:
                    continue
                p_line = {
                    'cnf': words[1],
                    'nbvar': int(words[2]),
                    'nbclauses': int(words[3])
                }
            else:
                for w in words:
                    if len(w) == 0:
                        continue
                    var = int(w)
                    if var == 0:
                        clauses.append(ConstraintClause(current_clause, len(clauses) + 1))
                        current_clause = []
                    else:
                        current_clause.append(var)
        cnf['p_line'] = p_line
        cnf['clauses'] = clauses
        return cnf

    def _get_text(self, nodelist):
        ''' helper function to extract text data from xml tag. '''
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data)
        return ''.join(rc)

    def _get_text_of_field(self, dom_elmt, field_name):
        ''' get text contained by dom_element under given nested field. '''
        return self._get_text(dom_elmt.getElementsByTagName(field_name)[0].childNodes)

    def _get_text_list_of_fields(self, dom_elmt, field_name):
        ''' get list of text contained by nested fields identified by field_name. '''
        str_list = []
        for e in dom_elmt.getElementsByTagName(field_name):
            str_list.append(self._get_text(e.childNodes))
        return str_list


if __name__ == '__main__':
    features, interactions, cnf = parse('src/project_public_1/bdbc')
    print('#############FEATURES##############')
    for key in features:
        print('=====', key, '=====\n', features[key])
    print('#############INTERACTIONS##############')
    for i in interactions:
        print('=====', i.feature_names, '=====\n', i)
