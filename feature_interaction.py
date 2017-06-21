class FeatureInteraction(object):
  def __init__(self, feature_names=[], value=0.0):
    self.feature_names=feature_names
    self._value=value

  def __repr__(self):
    return "<feature_interaction feature_names=%s value=%s>" % (self.feature_names, self._value)
  
  def get_value(self, features):
    for f_name in self.feature_names:
      if not f_name in features:
        return 0.0
    return self._value