from xml.dom import minidom
from feature import Feature
from feature_interaction import FeatureInteraction

def parse(feature_prefix, verbose=False):
  fp = FeatureParser(feature_prefix)
  features, interactions = fp.parse()
  if verbose:
    print '#############FEATURES##############'
    for key in features:
      print '==========\n', features[key]
    print '#############INTERACTIONS##############'
    for i in interactions:
      print '==========\n', i
  return features, interactions

class FeatureParser(object):
  def __init__(self, feature_prefix):
    self.feature_prefix = feature_prefix
    self.parsed_features = {}
    self.parsed_interactions = []
  
  def _create_dict_from_txt(self, path):
    """
    Creates a feature or interaction dict from the given txt file
    :param path: string
    :return: dict
    """
    # read in feature and interaction files
    with open(path, "r") as file:
        features = file.read().replace(" ", "").split('\n')
    # create dict out of the given features
    pairs = [feature.split(":") for feature in features if feature is not ""]
    feature_dict = dict((k.strip(), float(v.strip())) for k, v in pairs)
    return feature_dict
    
  def parse(self):
    ''' extracts a list of Feature objects from given xml file. 
        results will be returned as dictionary. 
        Dictonary is also stored by class (see FeatureParser.parsed_features).'''
    # parse xml doc to get xml_feature list
    xml_feature_list = self._get_xml_feature_list()
    
    # iterate over all xml_features and compile list of Feature objects
    feature_list = {}
    feature_excludes = {}
    feature_values = self._create_dict_from_txt(self.feature_prefix+"_feature.txt")
    for xml_feature in xml_feature_list:
      # extract name of feature and continue only if name is set
      f_name = self._get_text_of_field(xml_feature, 'name')
      if len(f_name) == 0:
        continue
      f_value = feature_values[f_name]
      # create Feature and set class properties
      f = Feature(f_name, f_value)
      f.output_string = self._get_text_of_field(xml_feature, 'outputString')
      f.default_value = self._get_text_of_field(xml_feature, 'defaultValue')
      f.optional = self._get_text_of_field(xml_feature, 'optional')
      # store string list of exclude features for later
      xml_exclude_options = xml_feature.getElementsByTagName('excludedOptions')[0]
      if len(xml_exclude_options.childNodes) > 0:
        feature_excludes[f_name] = self._get_text_list_of_fields(xml_exclude_options, 'options')
      else:
        feature_excludes[f_name] = []
      # append feature to list
      feature_list[f_name] = f
    
    # compile list of exclude_features for each Feature
    # given string dictionary extracted from xml
    for key in feature_list:
      if len(feature_excludes[key]) == 0:
        pass
      else:
        f = feature_list[key]
        f.exclude_features = [feature_list[f_name] for f_name in feature_excludes[key]]
    
    # compile list of feature_interactions
    interactions = self._create_dict_from_txt(self.feature_prefix+"_interactions.txt")
    self.parsed_interactions = [FeatureInteraction(key.split('#'),value) for key, value in interactions.items()]
    self.parsed_features = feature_list
    return self.parsed_features, self.parsed_interactions

  def _get_text(self, nodelist):
    ''' helper function to extract text data from xml tag. '''
    rc = []
    for node in nodelist:
      if node.nodeType == node.TEXT_NODE:
        rc.append(node.data)
    return ''.join(rc)

  def _get_text_of_field(self, dom_elmt, field_name):
    ''' get text contained by dom_element under given nested field. '''
    return self._get_text(dom_elmt.getElementsByTagName(field_name)[0].childNodes)

  def _get_text_list_of_fields(self, dom_elmt, field_name):
    ''' get list of text contained by nested fields identified by field_name. '''
    str_list = []
    for e in dom_elmt.getElementsByTagName(field_name):
      str_list.append(self._get_text(e.childNodes))
    return str_list
  
  def _get_xml_feature_list(self):
    ''' open xml file and extract all xml_feature objects. '''
    xml_doc = minidom.parse(self.feature_prefix+'_model.xml')
    return xml_doc.getElementsByTagName('configurationOption')
  
if __name__ == '__main__':
  features, interactions = parse('src/project_public_1/bdbc')
  print '#############FEATURES##############'
  for key in features:
    print '==========\n', features[key]
  print '#############INTERACTIONS##############'
  for i in interactions:
    print '==========\n', i