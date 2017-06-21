class Feature(object):
  def __init__(self, name, output_string='', default_value='Selected', optional=True, exclude_features=[]):
    self.name = name
    self.output_string = output_string
    if len(self.output_string) == 0:
      self.output_string = self.name
    self.default_value = default_value
    self.optional = optional
    self.exclude_features = exclude_features
  
  def __repr__(self):
    return "<feature name=%s output_string=%s default_value=%s optional=%s exclude_features=%s>" % (
        self.name, self.output_string, self.default_value, self.optional, self.exclude_features)