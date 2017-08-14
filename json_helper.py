import json
import os
import os.path
import datetime

def write_json_to_file(json_dict, file_name):
    ''' (over)writes contents of json_dict into file referenced by file_name. '''
    with open(file_name, 'w') as file:
        json.dump(json_dict, file, sort_keys=True, indent=4, separators=(',', ': '))
        file.close()

def extend_json_log(json_dict, file_name):
    ''' appends contents of json_dict to logging structure 
        in JSON file referenced by file_name. '''
    time_stamp = str(datetime.datetime.now())
    full_json = None
    if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0:
        full_json = {time_stamp: json_dict}
    else:
        with open(file_name, 'r') as file:
            full_json = json.load(file)
            file.close()
        full_json[time_stamp] = json_dict
    if 'best' in json_dict:
        if 'best' in full_json:
            if json_dict['best']['fitness'] < full_json['best']['fitness']:
                full_json['best'] = json_dict['best']
                full_json['best']['time_stamp'] = time_stamp
        else:
            full_json['best'] = json_dict['best']
            full_json['best']['time_stamp'] = time_stamp
    write_json_to_file(full_json, file_name)

def clear_json_log(file_name):
    write_json_to_file({}, file_name)