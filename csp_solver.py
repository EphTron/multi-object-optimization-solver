import copy
import random

GLOBAL_INSTANCE = None


class CSPSolver(object):
    def __init__(self, constraints):
        self.constraints = constraints
        self.primitive_constraints = None
        self.composite_constraints = None
        self.primitive_redundant_constraints = None
        self.constraint_graph = None
        self.feature_spaces = None
        self.evaluate_constraint_clauses()

    ########################################
    ############ INITIALIZATION ############
    ########################################

    def evaluate_constraint_clauses(self):
        ''' Initialization function to setup evaluation structures for CSP solving,
            given structure of constraint cnf clauses. Fills member variables. '''
        self.primitive_constraints = []
        self.primitive_redundant_constraints = {}
        temp = copy.deepcopy(self.constraints['clauses'])

        search = True
        while search:
            search = False
            kill_list = []
            for c in temp:
                if len(c.clause) == 1:
                    self.primitive_constraints.append(c.clause[0])
                    kill_list.append(c)
                    search = True
                else:
                    for i in range(0, len(c.clause)):
                        if c.clause[i] in self.primitive_constraints:
                            kill_list.append(c)
                            self.primitive_redundant_constraints[c.id] = c
                            break
            while len(kill_list) != 0:
                temp.remove(kill_list[-1])
                kill_list.pop()

        self.composite_constraints = temp

        self._compute_constraint_graph()
        self._compute_feature_spaces(self.get_unsolved_features())

        print("number of total constraints:", len(self.constraints['clauses']))
        print("number of primitive constraints:", len(self.primitive_constraints))
        print("number of primitive redundant constraints:", len(self.primitive_redundant_constraints))
        print("number of composite constraints:", len(self.composite_constraints))
        print("FEATURE SPACES:", len(self.feature_spaces))

    def _compute_constraint_graph(self):
        ''' Helper function for evaluate_constraint_clauses, to build the constraint graph. '''
        self.constraint_graph = {cnf_id: [] for cnf_id in self.constraints['cnf_id_to_f_name'].keys()}

        for c in self.composite_constraints:
            for var in c.clause:
                if c.id not in self.constraint_graph[abs(var)]:
                    self.constraint_graph[abs(var)].append(c.id)

        for i in self.constraint_graph:
            if len(self.constraint_graph[i]) != 0:
                print("==================")
                print("(feature", i, ") constraints:", self.constraint_graph[i])
                print("   ===== DETAILS =====")
                for c_id in self.constraint_graph[i]:
                    print(self.constraints['clauses'][c_id - 1])

    def _compute_feature_spaces(self, features):
        constraint_spaces = []
        for cnf_id in features:
            temp = self.constraint_graph[cnf_id]

            associated_spaces = []
            for space in constraint_spaces:
                for c_id in temp:
                    if c_id in space and space not in associated_spaces:
                        associated_spaces.append(space)

            # remove all associated spaces from constraint spaces
            # then merge all spaces into new space
            merged_space = []
            for space in associated_spaces:
                constraint_spaces.remove(space)
                merged_space.extend(space)
            merged_space.extend(temp)

            # append new space
            constraint_spaces.append(list(set(merged_space)))

        self.feature_spaces = []
        for c_space in constraint_spaces:
            self.feature_spaces.append([])
            print("==================\nFEATURE SPACE " + str(len(self.feature_spaces) - 1))
            for c_id in c_space:
                c = self.constraints['clauses'][c_id - 1]
                additional_features = [f_id for f_id in c.get_features() if f_id not in self.feature_spaces[-1]]
                self.feature_spaces[-1].extend(additional_features)
            print(" > FEATURES    :" + str(len(self.feature_spaces[-1])))
            print(" > CONSTRAINTS :" + str(len(c_space)))

    ##################### END INITIALIZATION #####################

    def generate_feature_vector(self):
        ''' Creates a valid feature vector,
            solving set constraints. '''
        result = {
            cnf_id: None for cnf_id in self.constraints['cnf_id_to_f_name'].keys()
        }

        result_tested = {
            cnf_id: [] for cnf_id in result.keys()
        }

        self.solve_primitive(result, result_tested)

        i = 0
        for space in self.feature_spaces:
            # print("Starting space", i, " (length:", len(space), ")")
            if not self._solve_feature_values(space, result, result_tested):
                print("Failure: Contradicting cnf File")
                return None
            i += 1

        return result

    def fix_feature_vector(self, vector):
        result = {
            cnf_id: vector[cnf_id] for cnf_id in self.constraints['cnf_id_to_f_name'].keys()
        }

        print("WHAAAAT UP", result)

        result_tested = {
            cnf_id: [] for cnf_id in result.keys()  ##[vector[cnf_id]] for cnf_id in result.keys()
        }

        self.solve_primitive(result, result_tested)

        i = 0
        for space in self.feature_spaces:
            print("Starting space", i, " (length:", len(space), ")")
            if not self._solve_feature_values(space, result, result_tested):
                print("Failure: Contradicting cnf File")
                return None
            i += 1

        return result

    def get_unsolved_features(self):
        ''' Returns all features which are not solved by primitive constraints. '''
        all_features = {
            cnf_id: None for cnf_id in self.constraints['cnf_id_to_f_name'].keys()
        }

        # set fields with primitive constraints
        unsolved = []
        for cnf_id in all_features:
            if cnf_id in self.primitive_constraints:
                pass
            elif -1 * cnf_id in self.primitive_constraints:
                pass
            else:
                unsolved.append(cnf_id)

        return unsolved

    def solve_primitive(self, result, result_tested):
        # set fields with primitive constraints
        useless = []
        for cnf_id in result:
            if cnf_id in self.primitive_constraints:
                result[cnf_id] = True
                result_tested[cnf_id] = [True, False]
                useless.append(cnf_id)
            elif -1 * cnf_id in self.primitive_constraints:
                result[cnf_id] = False
                result_tested[cnf_id] = [True, False]
                useless.append(cnf_id)
        a = sum([len(space) for space in self.feature_spaces])
        print("PRIMITIVE FEATURES:", useless)
        print("FEATURES added from spaces", self.feature_spaces)
        print("FEATURES sum", len(useless) + a)
        print("FEATURES expected", len(result))

    def _next_idx(self, feature_ids, current_i):
        ''' Helper function for generate_feaure_vector to determine index of next feature to be tested.
            Takes into account primitive_constraints. '''
        i = current_i + 1
        while i < len(feature_ids) and (
                feature_ids[i] in self.primitive_constraints or -1 * feature_ids[i] in self.primitive_constraints):
            i += 1
        # print("next", i)
        return i

    def _prev_idx(self, feature_ids, current_i):
        ''' Helper function for generate_feaure_vector to determine index of feature being tested previously.
            Takes into account primitive_constraints. '''
        i = current_i - 1
        while i >= len(feature_ids) and (
                feature_ids[i] in self.primitive_constraints or -1 * feature_ids[i] in self.primitive_constraints):
            i -= 1
        # print("prev", i)
        return i

    def _solve_feature_values(self, features, result, result_tested):
        ''' Recursively solves missing values for features in result identified by cnf_id's listed in features.
            result_tested stores information about previously tested values of a feature. '''
        i = 0
        cleared = False
        while i < len(features):
            if i < 0:
                return False

            cnf_id = features[i]

            # solve using random value for untested features
            if result[cnf_id] == None:

                v = random.randint(0, 1) > 0

                # pick most constrained value if feature has constraints
                if len(self.constraint_graph[cnf_id]) > 0:
                    c = self._pick_best_constraint(self.constraint_graph[cnf_id], result)
                    v = c.get_value(cnf_id)

                # set picked value as tested for given feature
                result_tested[cnf_id].append(v)
                if not self._solve(cnf_id, result, v, result_tested):
                    # test opposite if this didn't solve
                    result_tested[cnf_id].append(not v)
                    if not self._solve(cnf_id, result, not v, result_tested):
                        i = self._step_back(features, i, result, result_tested, not cleared)
                        cleared = True
                        continue

            # pick value from untested
            else:
                # solve using untested value
                if len(result_tested[cnf_id]) == 1:
                    v = not result_tested[cnf_id][0]
                    result_tested[cnf_id].append(v)
                    if not self._solve(cnf_id, result, v, result_tested):
                        i = self._step_back(features, i, result, result_tested, not cleared)
                        cleared = True
                        continue
                # backtrack if all values tested for feature id
                else:
                    i = self._step_back(features, i, result, result_tested, not cleared)
                    cleared = True
                    continue

            i = self._next_idx(features, i)
        return True

    def _step_back(self, features, i, result, result_tested, clear=False):
        ''' Clears tested values and set value for feature referenced by 
            cnf_id equal to features[i]. If clear equals True all values
            of features referenced by features identified by features[a],
            (where i < a < len(features)) will also be cleared.
            Returns self._prev_idx(features, i) of i. '''
        result_tested[features[i]] = []
        result[features[i]] = None
        if clear:
            self._clear_following_features(features, i, result, result_tested)
        return self._prev_idx(features, i)

    def _clear_following_features(self, features, i, result, result_tested):
        ''' clears all feature values and tested values for features following i in features. '''
        temp = i
        while temp < len(features):
            result[features[temp]] = None
            result_tested[features[temp]] = []
            temp = self._next_idx(features, temp)
        return

    def _pick_best_constraint(self, constraint_ids, result):
        ''' picks the best constraint given by how many terms in it have already been set. '''
        best_constraint = None
        for c_id in constraint_ids:
            if c_id - 1 in self.primitive_redundant_constraints:
                continue
            c = self.constraints['clauses'][c_id - 1]
            if c.is_solved_by_list(result):
                continue
            if best_constraint == None:
                best_constraint = c
                continue
            if len(best_constraint.get_culled_clause(result)) > len(c.get_culled_clause(result)):
                best_constraint = c

        if best_constraint != None:
            return best_constraint
            '''
            print "Constraint " + str(best_constraint.id)
            print " > Clause " + str(best_constraint.clause)
            print " > Culled " + str(best_constraint.get_culled_clause(result))
            print " > Picked Value for feature.id="+str(cnf_id)+" is "+str(v)
            '''
        else:
            idx = random.randint(0, len(self.constraint_graph[cnf_id]) - 1)
            c_id = constraint_ids[idx]
            return self.constraints['clauses'][c_id - 1]

    def _solve(self, cnf_id, result, value, result_tested):
        ''' Recursive Helper function for _satisfy_constraints, to solve the value for a single constraint. '''
        prev = result[cnf_id]
        result[cnf_id] = value

        unsatisfied = []
        for c_id in self.constraint_graph[cnf_id]:
            if self.constraints['clauses'][c_id - 1].is_violated_by(result):
                result[cnf_id] = prev
                return False
                # elif not self.constraints['clauses'][c_id-1].is_solved_by_list(result):
                #  unsatisfied.append(c_id)

        return True  # self._satisfy_constraints(unsatisfied, result, result_tested)

    def _satisfy_constraints(self, unsatisfied, result, result_tested):
        pass
