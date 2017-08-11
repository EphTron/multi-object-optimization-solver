class ConstraintClause(object):
  def __init__(self, clause, id):
    self.clause = clause
    self.id = id
    self.satisfied = False
  
  def is_met_by(self, cnf_ids):
    for var in self.clause:
      if var < 0 and abs(var) not in cnf_ids:
        return True
      elif var > 0 and var in cnf_ids:
        return True
    return False  

  def is_solved_by(self, cnf_id, is_set):
    if cnf_id not in [abs(var) for var in self.clause]:
      return False

    if is_set:
      return cnf_id in self.clause
    return -1*cnf_id in self.clause
  
  def is_violated_by(self, feature_vector):
    for var in self.clause:
      if feature_vector[abs(var)] == None:
        return False
      elif var < 0 and feature_vector[abs(var)] == False:
        return False
      elif var > 0 and feature_vector[var] == True:
        return False
    return True
  
  def get_culled_clause(self, feature_vector):
    ''' returns the part of the clause could still be met
    by the feature vector without flipping set boolean values. '''
    result = []
    for var in self.clause:
      if feature_vector[abs(var)] == None:
        result.append(var)
    return result
  
  def __repr__(self):
    return "<ConstraintClause id=%s clause=%s satisfied=%s>" % (self.id, self.clause, self.satisfied)