import random

def generate_random(feature_dict={}, ensure_valid=True):
    ''' Brute force generation of valid candidate solution.
    feature_dict contains ALL possible features. '''
    candidate = None
    while candidate == None:
        f_dict = {}
        for key in feature_dict:
            if random.randint(0,1) == 1:
                f_dict[key] = feature_dict[key]
            else:
                f_dict[key] = None
        candidate = CandidateSolution(f_dict)
        if ensure_valid and not candidate.is_valid():
            candidate = None
    return candidate

def assess_fitness(candidate, interactions=None):
    features = candidate.get_feature_list()
    feature_fitness = 0
    for feature in features:
        feature_fitness += feature.value

    interaction_fitness = 0
    if interactions != None:
        for i in interactions:
            interaction_fitness += i.get_value(features)

    return feature_fitness + interaction_fitness
    
def arbitrary_crossover(p1, p2, ensure_valid=True):
    c1 = None
    c2 = None
    while c1 is None or c2 is None:
        c1_features = p1.get_feature_dict()
        c2_features = p2.get_feature_dict()
        for f_name in c1_features.keys():
            if random.randint(0,1) == 1:
                temp = c1_features[f_name]
                c1_features[f_name] = c2_features[f_name]
                c2_features[f_name] = temp
        c1 = CandidateSolution(c1_features)
        c2 = CandidateSolution(c2_features)
        if ensure_valid and (not c1.is_valid() or not c2.is_valid()):
            c1 = None
            c2 = None
    return c1, c2
            
class CandidateSolution:
    ''' Contains a configuration of features in a dict.
    Unset features have a None value for specific key. '''
    
    interactions = None
    
    def __init__(self, features={}):
        self._features = features
        self._feature_list = [f for f in self._features.values() if f is not None]
        self._fitness = None
        if CandidateSolution.interactions != None:
            self.calc_fitness()

    def calc_fitness(self):
        self._fitness = assess_fitness(self, CandidateSolution.interactions)
    
    def get_fitness(self):
        return self._fitness
        
    def is_valid(self):
        ''' checks constraints in feature list.
        Returns True if all constraints met. '''
        for feature in self._features.values():
            if feature == None:
                continue
            for ex_feature in feature.exclude_features:
                if ex_feature in self._features.values():
                    return False
        return True
    
    def get_feature_list(self):            
        return self._feature_list
        
    def get_feature_dict(self):            
        return self._features
        
    def copy_from(self, c):
        ''' deep copy of candidate_solution data. '''
        self._features = {f_name:c._features[f_name] for f_name in c._features}
        self._feature_list = [f for f in self._features.values() if f is not None]