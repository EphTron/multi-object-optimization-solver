import copy
import random
    
GLOBAL_INSTANCE = None

class CSPSolver(object):
  def __init__(self, constraint_cnf):
    self.constraint_cnf = constraint_cnf
    self.primitive_constraints = None
    self.composite_constraints = None
    self.primitive_redundant_constraints = None
    self.constraint_graph = None
    self.evaluate_constraint_clauses()
    
  def generate_feature_vector(self):
    # initialize result with all unset fields
    for c in self.constraint_cnf['clauses']:
      c.satisfied = False
      
    result = {cnf_id: None for cnf_id in self.constraint_cnf['cnf_id_to_f_name'].keys()}    
    
    # set fields with primitive constraints
    done = True
    for cnf_id in result:
      if cnf_id in self.primitive_constraints:
          result[cnf_id] = True
      elif -1*cnf_id in self.primitive_constraints:
          result[cnf_id] = False
      else:
        done = False
    
    for cnf_id in result:
      if result[cnf_id] == None:
        print("tyring to solve", cnf_id)
        if not self.solve(cnf_id, result):
          print "Failure: Contradicting cnf File"
          print " > Couldn't solve for:", cnf_id
          return None
          
    return result
    
  def solve(self, cnf_id, result, default=None):
    if default == None:
      d = random.randint(0,1) > 0
      if not self.solve(cnf_id, result, default=d):
        return self.solve(cnf_id, result, default=not d)
      return True
    
    prev = result[cnf_id]    
    result[cnf_id] = default
      
    satisfied_constraints = []
    all_solved = True
    for c_id in self.constraint_graph[cnf_id]:
      if self.constraint_cnf['clauses'][c_id-1].satisfied:
        continue
      if self.constraint_cnf['clauses'][c_id-1].is_solved_by(cnf_id, result[cnf_id]):
        satisfied_constraints.append(c_id)
        self.constraint_cnf['clauses'][c_id-1].satisfied = True
      elif self.constraint_cnf['clauses'][c_id-1].is_violated_by(result):
        while len(satisfied_constraints) > 0:
          self.constraint_cnf['clauses'][satisfied_constraints[-1]].satisfied = False
          satisfied_constraints.pop()
        result[cnf_id] = prev
        print "Fail 1"
        return False
      else: 
        all_solved = False
    
    while not all_solved:
      all_solved = True
      one_solved = False
      for c_id in self.constraint_graph[cnf_id]:
        c = self.constraint_cnf['clauses'][c_id-1]
        if not c.satisfied:
          culled_clause = c.get_culled_clause(result)
          if len(culled_clause) == 1:
            var = culled_clause[0]
            if not self.solve(abs(var), result, var > 0):
              while len(satisfied_constraints) > 0:
                self.constraint_cnf['clauses'][satisfied_constraints[-1]].satisfied = False
                satisfied_constraints.pop()
              result[cnf_id] = prev
              print "Fail 2"
              return False
            satisfied_constraints.append(c_id)
            c.satisfied = True
            one_solved = True
          else:
            all_solved = False
      if not one_solved:
        for c_id in self.constraint_graph[cnf_id]:
          c = self.constraint_cnf['clauses'][c_id-1]
          if not c.satisfied:
            culled_clause = c.get_culled_clause(result)
            for var in culled_clause:
              if self.solve(abs(var), result, var > 0):
                one_solved = True
                break
        if not one_solved:
          while len(satisfied_constraints) > 0:
            self.constraint_cnf['clauses'][satisfied_constraints[-1]].satisfied = False
            satisfied_constraints.pop()
          result[cnf_id] = prev
          print "Fail 3"
          return False
    
    return True
  
  def evaluate_constraint_clauses(self):
    self.primitive_constraints = []
    self.primitive_redundant_constraints = []
    temp = copy.deepcopy(self.constraint_cnf['clauses'])
    
    '''
    search = True
    while search:
      search = False
      kill_list = []
      for c in temp:
        if len(c.clause) == 1:
          self.primitive_constraints.append(c.clause[0])
          kill_list.append(c)
          search=True
        else:
          for i in range(0, len(c.clause)):
            if c.clause[i] in self.primitive_constraints:
              kill_list.append(c)
              self.primitive_redundant_constraints.append(c)
              break
      while len(kill_list) != 0:
        temp.remove(kill_list[-1])
        kill_list.pop()
    '''
    
    self.composite_constraints = temp

    self._compute_constraint_graph()
    
    print("number of total constraints:", len(self.constraint_cnf['clauses']))
    print("number of primitive constraints:", len(self.primitive_constraints))
    print("number of primitive redundant constraints:", len(self.primitive_redundant_constraints))
    print("number of composite constraints:", len(self.composite_constraints))
    
  def _compute_constraint_graph(self):
    self.constraint_graph = {cnf_id: [] for cnf_id in self.constraint_cnf['cnf_id_to_f_name'].keys()}

    for c in self.composite_constraints:
      for var in c.clause:
        if c.id not in self.constraint_graph[abs(var)]:
          self.constraint_graph[abs(var)].append(c.id)
    
    for i in self.constraint_graph:
      if len(self.constraint_graph[i]) != 0:
        print("==================")
        print("(feature", i ,") constraints:", self.constraint_graph[i])
        print("   ===== DETAILS =====")
        for c_id in self.constraint_graph[i]:
          print(self.constraint_cnf['clauses'][c_id-1])          