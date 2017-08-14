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
        self._values = [value]
    
    def get_value(self, i):
        ''' Returns the ith value of this feature. Number of values 
            depends on values added to the feature. '''
        if i < len(self._values):
            return self._values[i]
        else:
            msg = "Failure: Feature.get_value(i): i not within range"
            msg += "\n > expected: i < " + str(len(self._values))
            msg += "\n > received: " + str(i)
            raise ValueError("Failure: Feature.get_value(i): i not within range")
    
    def add_value(self, v):
        ''' appends a value to this feature. '''
        self._values.append(v)
    
    def get_value_count(self):
        ''' Returns the number of available values for this feature. '''
        return len(self._values)
    
    def remove_value(self, i):
        ''' Removes the ith value of this feature. Number of values 
            depends on values added to the feature. '''
        if i < len(self._values):
            self._values.pop(i)
        else:
            msg = "Failure: Feature.remove_value(i): i not within range"
            msg += "\n > expected: i < " + str(len(self._values))
            msg += "\n > received: " + str(i)
            raise ValueError("Failure: Feature.get_value(i): i not within range")

    def __repr__(self):
        return "<feature name=%s values=%s cnf_id=%s output_string=%s default_value=%s optional=%s exclude_features=%s>" % (
            self.name, self._values, self.cnf_id, self.output_string, self.default_value, self.optional,
            self.exclude_features)
