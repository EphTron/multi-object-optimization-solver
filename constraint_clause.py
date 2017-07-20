class ConstraintClause(object):
  def __init__(self, clause):
    self.clause = clause
  
  def is_met_by(self, cnf_ids):
    for var in self.clause:
      if var < 0 and abs(var) not in cnf_ids:
        return True
      elif var > 0 and var in cnf_ids:
        return True
    return False  
    
  def __repr__(self):
    return "<ConstraintClause clause=%s>" % (self.clause)