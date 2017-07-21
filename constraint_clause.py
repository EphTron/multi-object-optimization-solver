class ConstraintClause(object):
  def __init__(self, clause, id):
    self.clause = clause
    self.id = id
  
  def is_met_by(self, cnf_ids):
    for var in self.clause:
      if var < 0 and abs(var) not in cnf_ids:
        return True
      elif var > 0 and var in cnf_ids:
        return True
    return False  
    
  def __repr__(self):
    return "<ConstraintClause id=%s clause=%s>" % (self.id, self.clause)