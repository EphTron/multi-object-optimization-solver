import random
import csp_solver


def generate_random(feature_dict={}, ensure_valid=True):
    ''' Brute force generation of valid candidate solution.
    feature_dict contains ALL possible features. '''
    candidate = None
    while candidate == None:
        f_dict = {}
        for key in feature_dict:
            gen_random = False
            if csp_solver.GLOBAL_INSTANCE != None:
                f = feature_dict[key]
                if f.cnf_id == None:
                    gen_random = True
                else:
                    if f.cnf_id in csp_solver.GLOBAL_INSTANCE.primitive_constraints:
                        f_dict[key] = feature_dict[key]
                    elif -1 * f.cnf_id in csp_solver.GLOBAL_INSTANCE.primitive_constraints:
                        f_dict[key] = None
                    else:
                        gen_random = True
            else:
                gen_random = True
            if gen_random:
                if random.randint(0, 1) == 1:
                    f_dict[key] = feature_dict[key]
                else:
                    f_dict[key] = None
        candidate = CandidateSolution(f_dict)
        if ensure_valid and not candidate.is_valid():
            candidate = None
    return candidate


def assess_fitness(candidate, objective_idx):
    features = candidate.get_features()
    feature_fitness = 0
    for feature_id, is_set in features.items():
        if is_set:
            feature = CandidateSolution.model.get_feature_by_id(feature_id)
            feature_fitness += feature.get_value(objective_idx)

    interactions = CandidateSolution.model.get_interaction_set(objective_idx)
    interaction_fitness = 0
    if interactions != None:
        for i in interactions:
            interaction_fitness += i.get_value(features)

    # print("Calculated feature fitness" , feature_fitness)

    return feature_fitness + interaction_fitness  # TODO comment back in


def arbitrary_crossover(p1, p2, ensure_valid=True):
    c1 = None
    c2 = None
    while c1 is None or c2 is None:
        c1_features = p1.get_feature_dict()
        c2_features = p2.get_feature_dict()
        for f_name in c1_features.keys():
            if random.randint(0, 1) == 1:
                temp = c1_features[f_name]
                c1_features[f_name] = c2_features[f_name]
                c2_features[f_name] = temp
        c1 = CandidateSolution(c1_features)
        c2 = CandidateSolution(c2_features)
        if ensure_valid and (not c1.is_valid() or not c2.is_valid()):
            c1 = None
            c2 = None
    return c1, c2


def get_feature_cost(id, objective_id=0):
    # if CandidateSolution.min_feature_value == None:
    # TODO: list
    feature_values = CandidateSolution.model.get_all_feature_values(objective_id)
    CandidateSolution.min_feature_value = min(feature_values)
    CandidateSolution.max_feature_value = max(feature_values)
    feature = CandidateSolution.model.get_feature_by_id(id)
    v = feature.get_value(objective_id)
    return map_to_range(v, CandidateSolution.min_feature_value, CandidateSolution.max_feature_value, 1, 100)


def map_to_range(value=-0.3, old_min=-0.5, old_max=0.5, new_min=0, new_max=1):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((value - old_min) * new_range) / old_range) + new_min
    return new_value


class CandidateSolution:
    ''' Contains a configuration of features in a dict.
    Unset features have a None value for specific key. '''

    model = None
    number_of_instances = 0
    max_feature_value = None
    min_feature_value = None

    def __init__(self, features={}):
        self._id = CandidateSolution.number_of_instances
        CandidateSolution.number_of_instances += 1

        self._features = features
        self._feature_list = [f for f in self._features.values() if f is not None]
        self._fitness_values = []
        self.pareto_rank = None
        self.sparsity = 0
        if CandidateSolution.model != None:
            self.calc_fitness()

    def get_id(self):
        return self._id

    def calc_fitness(self):
        self._fitness_values = []
        for i in range(0, CandidateSolution.model.get_num_objectives()):
            self._fitness_values.append(assess_fitness(self, i))

    def get_fitness(self, objective_idx):
        ''' Returns the candidate solutions fitness 
            for objective referenced by objective_idx. '''
        if objective_idx >= len(self._fitness_values):
            raise ValueError("Failure: Exceedes value range for objectives.")
        return self._fitness_values[objective_idx]

    def get_fitness_values(self):
        ''' Returns a list of all objective fitness values
            for this candidate. '''
        return [v for v in self._fitness_values]

    def get_fitness_sum(self):
        return sum(self.get_fitness_values())

    def is_valid(self):
        ''' checks constraints in feature list.
        Returns True if all constraints met. '''
        cnf_ids = []
        for feature in self._features.values():
            if feature == None:
                continue
            if feature.cnf_id != None:
                cnf_ids.append(feature.cnf_id)
            # evaluate exclude features in case model.xml was used
            for ex_feature in feature.exclude_features:
                if ex_feature in self._features.values():
                    return False
        return True

    def get_feature_list(self):
        return self._feature_list

    def get_feature_dict(self):
        return self._features

    def get_features(self):
        return self._features

    def copy_from(self, c):
        ''' deep copy of candidate_solution data. '''
        self._features = {f_name: c._features[f_name] for f_name in c._features}
        self._feature_list = [f for f in self._features.values() if f is not None]

    def dominates(self, other):
        ''' returns true if this candidate dominates other candidate. 
            (implements PARETO dominance). '''
        epsilon = 0.0000000001
        all_equal = True
        for i in range(0, len(self._fitness_values)):
            this_val = self._fitness_values[i]
            other_val = other._fitness_values[i]
            if abs(this_val - other_val) < epsilon:
                continue
            elif this_val > other_val:
                return False
            else:
                all_equal = False
        if all_equal:
            return False
        return True

    def as_dict(self):
        ''' format all class attributes into a dict.
            Used for generating JSON output. '''
        return {
            'id': self._id,
            'fitness_values': self.get_fitness_values(),
            'features': {
                CandidateSolution.model.get_feature_name(id): val
                for id, val in self.get_features().items()
            }
        }
