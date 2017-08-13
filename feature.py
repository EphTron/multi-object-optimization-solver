class Feature(object):
    def __init__(self, name, value, output_string='', default_value='Selected', optional=True, exclude_features=[]):
        self.name = name
        self.cnf_id = None
        self.output_string = output_string
        if len(self.output_string) == 0:
            self.output_string = self.name
        self.default_value = default_value
        self.optional = optional
        self.exclude_features = exclude_features
        self.value = value

    def __repr__(self):
        return "<feature name=%s value=%s cnf_id=%s output_string=%s default_value=%s optional=%s exclude_features=%s>" % (
            self.name, self.value, self.cnf_id, self.output_string, self.default_value, self.optional,
            self.exclude_features)
