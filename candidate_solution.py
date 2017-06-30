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
    return CandidateSolution(f_dict)

class CandidateSolution:
    ''' Contains a configuration of features in a dict.
    Unset features have a None value for specific key. '''
    
    def __init__(self, features={}):
        self._features = features
        self._feature_list = [f for f in self._features.values() if f is not None]
    
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