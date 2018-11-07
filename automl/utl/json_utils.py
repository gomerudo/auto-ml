import json
import os
from ConfigSpace.read_and_write.json import read, write
from automl.globalvars import ROOT_DIR


def _convert_cs_to_json(cs):
    cs_as_string = write(cs)
    cs_as_json = json.loads(cs_as_string)
    return cs_as_json


def _convert_json_to_cs(json_obj):
    json_as_string = json.dumps(json_obj)
    json_as_cs = read(json_as_string)
    return json_as_cs


def _save_json_file(json_obj, json_name):
    path = ROOT_DIR+"/automl/createconfigspacepipeline/json_files_for_all_components/"+json_name+".json"
    with open(path, 'w') as outfile:
        json.dump(json_obj, outfile)


def write_cs_to_json_file(cs, json_name):
    """This function writes the configuration space object into a json file.

    Args:
        cs (ConfigurationSpace): The configuration space object
        json_name (str): Name of the component. The name of the component should be exactly same as the name of the
            class of the component.

    """
    json_obj = _convert_cs_to_json(cs)
    _save_json_file(json_obj, json_name)


def _read_json_file_to_json_obj(json_name):
    file_path = ROOT_DIR+"/automl/createconfigspacepipeline/json_files_for_all_components/"+json_name+".json"
    with open(file_path) as json_data:
        json_obj = json.load(json_data)
    return json_obj


def _check_existence(json_name):
    dir_path = ROOT_DIR+"/automl/createconfigspacepipeline/json_files_for_all_components/"
    file_name = json_name+".json"
    exist = os.path.isfile(dir_path+file_name)
    return exist
