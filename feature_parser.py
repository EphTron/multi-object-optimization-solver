from xml.dom import minidom
from feature import Feature
from constraint_clause import ConstraintClause
from feature_interaction import FeatureInteraction
import os.path

def parse(path_prefix, feature_path="", interaction_path="", model_path="", cnf_path="", verbose=False):
    fp = FeatureParser(path_prefix, feature_path, interaction_path, model_path, cnf_path)
    features, interactions, cnf = fp.parse()
    if verbose:
        print('#############FEATURES##############')
        for key in features:
            print(features[key])
        print('#############INTERACTIONS##############')
        for i in interactions:
            print(i)
        print('#############CNF CLAUSES#############')
        print(' > p_line:', cnf['p_line'])
        print(' > clauses:')
        for c in cnf['clauses']:
            print(c)
    return features, interactions, cnf


class FeatureParser(object):
    def __init__(self, path_prefix, feature_path="", interaction_path="", model_path="", cnf_path=""):
        self.path_prefix = path_prefix
        self.feature_path = feature_path
        if len(self.feature_path) == 0:
            self.feature_path = self.path_prefix + "_feature.txt"
        self.interaction_path = interaction_path
        if len(self.interaction_path) == 0:
            self.interaction_path = self.path_prefix + "_interactions.txt"
        self.model_path = model_path
        if len(self.model_path) == 0:
            self.model_path = self.path_prefix + "_model.xml"
        self.cnf_path = cnf_path
        self.parsed_features = {}
        self.parsed_interactions = []
        self.parsed_cnf = { 'p_line':None, 'clauses':[] }

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

    def parse(self):
        ''' extracts a list of Feature objects from given xml file. 
            results will be returned as dictionary. 
            Dictonary is also stored by class (see FeatureParser.parsed_features).'''
        # compile list of Feature objects from txt
        features = {
            name: Feature(name, value) 
                for name, value in self._create_dict_from_txt(self.feature_path).items()
        }
        
        # compile list of FeatureInteractions from txt
        interactions = [
            FeatureInteraction(key.split('#'), value) 
                for key, value in self._create_dict_from_txt(self.interaction_path).items()
        ]
        
        # check to see if xml model exists and extend features if found
        if os.path.isfile(self.model_path):
            self._parse_xml_model(features, self.model_path)

        # check to see if cnf file exists and extend features if found
        if os.path.isfile(self.cnf_path):
            self._parse_cnf(features, self.cnf_path)        

        self.parsed_interactions = interactions
        self.parsed_features = features
        return self.parsed_features, self.parsed_interactions, self.parsed_cnf
    
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
    
    def _parse_cnf(self, features, f_name):
        ''' HELPER function for parse.
            Parses a DIMACS CNF file to extend features with cnf_id
            and returns the list of clauses extracted from file. '''
        # read in feature and interaction files
        with open(f_name, "r") as file:
            # remove '$ ' symbols preceeding feature names
            lines = file.read().replace("$ ", " ").split("\n")
        clauses = []
        p_line = {'cnf':'','nbvar':0,'nbclauses':0}
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
            elif words[0] == "p":
                if len(words) != 4:
                    continue
                p_line = {
                    'cnf' : words[1],
                    'nbvar' : int(words[2]),
                    'nbclauses' : int(words[3])
                }
            else:
                for w in words:
                    if len(w) == 0:
                        continue
                    var = int(w)
                    if var == 0:
                        clauses.append(ConstraintClause(current_clause))
                        current_clause = []
                    else:
                        current_clause.append(var)
        self.parsed_cnf['p_line'] = p_line
        self.parsed_cnf['clauses'] = clauses

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
    features, interactions = parse('src/project_public_1/bdbc')
    print('#############FEATURES##############')
    for key in features:
        print('=====', key, '=====\n', features[key])
    print('#############INTERACTIONS##############')
    for i in interactions:
        print('=====', i.feature_names, '=====\n', i)
