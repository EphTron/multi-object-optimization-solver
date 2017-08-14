class FeatureModel(object):
    def __init__(self):
        self._features = {}
        self._interaction_sets = []
        self._cnf = {}
    
    def add_interaction_set(self, interactions):
        ''' Appends a set of interactions to list of interaction sets. '''
        self._interaction_sets.append(interactions)
    
    def get_interaction_set(self, i):
        ''' Returns the ith interaction set. '''
        if i < len(self._interaction_sets):
            return self._interaction_sets[i]
        else:
            raise ValueError("Failure: get_interaction_set(i): index not within range.")
    
    def get_num_interaction_sets(self):
        ''' Returns the total count of interactions. '''
        return len(self._interaction_sets)
    
    def set_features(self, features):
        self._features = features
    
    def get_features(self):
        return self._features
    
    def get_all_feature_values(self, i):
        ''' Returns a list of all of the features ith value. '''
        return [f.get_value(i) for f in self._features.values()]
    
    def get_feature_by_id(self, cnf_id):
        ''' Returns the feature corresponding to given cnf_id,
            or None if not found. '''
        if self._cnf == None:
            return None
        if cnf_id not in self._cnf["cnf_id_to_f_name"]:
            return None
        feature_name = self._cnf["cnf_id_to_f_name"][cnf_id]
        return self._features[feature_name]

    def set_cnf(self, cnf):
        self._cnf = cnf

    def get_cnf(self):
        return self._cnf