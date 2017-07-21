import copy
    
GLOBAL_INSTANCE = None

class CSPSolver(object):
  def __init__(self, constraint_cnf):
    self.constraint_clauses = constraint_cnf['clauses']
    self.p_line = constraint_cnf['p_line']
    self.primitive_constraints = None
    self.composite_constraints = None
    self.primitive_redundant_constraints = None
    self.constraint_graph = None
    self.evaluate_constraint_clauses()
  
  def evaluate_constraint_clauses(self):
    self.primitive_constraints = []
    self.primitive_redundant_constraints = []
    temp =  copy.deepcopy(self.constraint_clauses)
    
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
    
    self.composite_constraints = temp

    self._compute_constraint_graph()
    
    print("number of total constraints:", len(self.constraint_clauses))
    print("number of primitive constraints:", len(self.primitive_constraints))
    print("number of primitive redundant constraints:", len(self.primitive_redundant_constraints))
    print("number of composite constraints:", len(self.composite_constraints))
    
  def _compute_constraint_graph(self):
    self.constraint_graph = [[] for i in range(0, self.p_line['nbvar']+1)]

    for c in self.composite_constraints:
      for var in c.clause:
        if c.id not in self.constraint_graph[abs(var)]:
          self.constraint_graph[abs(var)].append(c.id)
    
    for i in range(1, len(self.constraint_graph)):
      if len(self.constraint_graph[i]) != 0:
        print("==================")
        print("(feature", i ,") constraints:", self.constraint_graph[i])
        print("   ===== DETAILS =====")
        for c_id in self.constraint_graph[i]:
          print(self.constraint_clauses[c_id-1])
    
          