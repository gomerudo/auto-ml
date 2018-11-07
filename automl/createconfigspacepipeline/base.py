from smac.configspace import ConfigurationSpace
from automl.utl import json_utils


class ConfigSpacePipeline:
    """This class deals with the creation and manipulation of configuration space from the given input pipeline.

    In addition to getting the configuration spaces from the predefined json files, it also resets the default values
    of the configuration space depending on the provided input. The upper and lower limit of a hyperparameter in a
    configuration space is also adjusted if the input hyperparameter value is out of range. This class is not supposed
    to be instantiated by the user instead is used by the BayesianOptimizationPipeline class.

    Args:
        pipeline (Pipeline): The pipeline for which configuration space will be generated.

    """

    def __init__(self, pipeline):
        """Initializer of the class ConfigSpacePipeline"""
        self.pipeline = pipeline

        self.combined_configuration_space = ConfigurationSpace()

    def get_config_space(self):
        """This function is used to get the combined configuration space for the pipeline.

        This function process each component in the pipeline, thereby adding the configuration space with reset
        default of each of the component and obtaining the complete configuration space. It adds configuration space of
        components within 'FeatureUnion' and deals with situation such as 'FeatureUnion' within 'FeatureUnion'.
        It also adds the configuration space of estimators of a component but not of the component itself.

        Returns:
            ConfigurationSpace: Configuration space of the pipeline.

        """
        for i in range(0, len(self.pipeline.steps)):
            self._process_component_config_space(self.pipeline.steps[i][1])

        return self.combined_configuration_space

    def _process_component_config_space(self, component):
        """Processes component's configuration space and adds it to the combined configuration space

        This functions also handles component such as 'FeatureUnion' by recursively adding the sub-component's
        configuration space. It also adds the configuration space of estimators of a component but not of the component
        itself.

        Args:
            component (obj): Object of the component

        """

        if self._get_component_name(component) == "FeatureUnion":
            for each_feature in component.transformer_list:
                self._process_component_config_space(each_feature[1])
        else:
            component_dict = component.get_params()
            if "estimator" in component_dict:
                component = component_dict["estimator"]
                component_dict = component.get_params()
            component_name = self._get_component_name(component)
            if self._component_json_exist(component_name):
                component_json = self._get_component_json(component_name)
                component_new_json = self._component_reset_default(component_json, component_dict)
                component_config_space = json_utils._convert_json_to_cs(component_new_json)
                component_number = self._component_already_exists(component_name, self.combined_configuration_space)
                component_name = component_name + "-" + str(component_number)
                # The following line adds two configuration space by adding a prefix of the component's name
                self.combined_configuration_space.add_configuration_space(component_name, component_config_space)

    @staticmethod
    def _get_component_json(component_name):
        """This function is used to get individual configuration space of a component in JSON format.

        Args:
            component_name (string): Name of the component as a string.

        Returns:
            dict: Individual configuration space in JSON format.

        """
        component_json = json_utils._read_json_file_to_json_obj(component_name)
        return component_json

    @staticmethod
    def _component_json_exist(component_name):
        """This function checks whether the configuration space of a component exits or not.

        Args:
            component_name (string): Name of the component as a string.

        Returns:
            bool: True if the component exists and False if it does noName of the component as a string.

        """
        exist = True if (json_utils._check_existence(component_name)) else False
        return exist

    def _component_reset_default(self, component_json, component_dict):
        """This function is used to reset the defaults of the config space of the component of the input pipeline.

        In addition to resetting the defaults, it also updates the upper and lower limit of input hyperparameter if its
        value is out of range.

        Note:
            Values for some hyperparameter in the configuration space are set to constant in order to optimize the
            result.
            Hyperparameter with variable input type (int, float, string) are set to the type which has been defined in
            the initial configuration space.

        Args:
            component_json (dict): Json of the component obtained from the pre-defined json file
            component_dict (dict): Dictionary of hyperparameters of the component in the input pipeline

        Returns:
            dict: Configuration space in JSON format with reset defaults.

        """
        for i in range(0, len(component_json['hyperparameters'])):
            if self._is_key_in_dic(component_dict, component_json['hyperparameters'][i]['name']):
                if self._is_key_in_dic(component_json['hyperparameters'][i], "default"):
                    if self._is_type_same(component_json['hyperparameters'][i]['default'],
                                          component_dict[component_json['hyperparameters'][i]['name']]):
                        if component_json['hyperparameters'][i]['type'] == "categorical":
                            component_json['hyperparameters'][i] = \
                                self._json_process_for_categorical(
                                    component_json['hyperparameters'][i],
                                    component_dict[component_json['hyperparameters'][i]['name']]
                                )
                        elif component_json['hyperparameters'][i]['type'] == "uniform_int":
                            component_json['hyperparameters'][i] = \
                                self._json_process_for_int_and_float(
                                    component_json['hyperparameters'][i],
                                    component_dict[component_json['hyperparameters'][i]['name']]
                                )
                        elif component_json['hyperparameters'][i]['type'] == "uniform_float":
                            component_json['hyperparameters'][i] = \
                                self._json_process_for_int_and_float(
                                    component_json['hyperparameters'][i],
                                    component_dict[component_json['hyperparameters'][i]['name']]
                                )
                    elif self._is_string_boolean(component_json['hyperparameters'][i]['default'],
                                                 component_dict[component_json['hyperparameters'][i]['name']]):
                        component_json['hyperparameters'][i]['default'] = \
                            str(component_dict[component_json['hyperparameters'][i]['name']])
                    elif self._is_string_none(component_json['hyperparameters'][i]['default'],
                                              component_dict[component_json['hyperparameters'][i]['name']]):
                        component_json['hyperparameters'][i]['default'] = \
                            str(component_dict[component_json['hyperparameters'][i]['name']])

        return component_json

    @staticmethod
    def _is_key_in_dic(component_dict, key):
        if key in component_dict:
            return True
        else:
            return False

    @staticmethod
    def _is_type_same(hyperparameter_1, hyperparameter_2):
        if isinstance(hyperparameter_1, type(hyperparameter_2)):
            return True
        else:
            return False

    @staticmethod
    def _json_process_for_categorical(hyperparameter_dict, value):
        """This function resets the default value of a categorical hyperparameter.

        Args:
            hyperparameter_dict (dict): Dictionary of hyperparameter name and the available choices.
            value (string): Value of the hyperparameter that we want to reset

        Returns:
            Dictionary with reset default value.

        """
        if value in hyperparameter_dict['choices']:
            hyperparameter_dict['default'] = value
        return hyperparameter_dict

    @staticmethod
    def _json_process_for_int_and_float(hyperparameter_dict, value):
        """This function resets the default value and (if necessary) upper-lower limit of int-float type hyperparameter.

        Args:
            hyperparameter_dict (string): Dictionary of hyperparameter name and the upper-lower limit.
            value (int or float): Value of the hyperparameter that we want to reset.

        Returns:
            dict: Dictionary with reset default value and upper-lower limit.

        """
        # Due to bug in ConfigSpace package the default value cannot be set lesser than 1e-10 and greater than 1e298
        # https: // github.com / automl / ConfigSpace / issues / 97
        if value < 1e-10 or value > 1e298:
            value = hyperparameter_dict['default']
        else:
            hyperparameter_dict['default'] = value
        if hyperparameter_dict['lower'] > value:
            hyperparameter_dict['lower'] = value
        elif hyperparameter_dict['upper'] < value:
            hyperparameter_dict['upper'] = value
        return hyperparameter_dict

    @staticmethod
    def _is_string_boolean(hyperparameter_1, hyperparameter_2):
        if ((hyperparameter_1 == "True" or hyperparameter_1 == "False") and
                (hyperparameter_2 is True or hyperparameter_2 is False)):
            return True
        else:
            return False

    @staticmethod
    def _is_string_none(hyperparameter_1, hyperparameter_2):
        if hyperparameter_1 == "None" and hyperparameter_2 is None:
            return True
        else:
            return False

    @staticmethod
    def _component_already_exists(component_name, combined_configuration_space):
        component_number = 0
        combined_configuration_space_json = json_utils._convert_cs_to_json(combined_configuration_space)
        for hyperparameter in combined_configuration_space_json['hyperparameters']:
            if hyperparameter['name'].startswith(component_name):
                c_full_name, h_name = hyperparameter['name'].split(':')
                c_name, c_number = c_full_name.split('-')
                if component_number < int(c_number):
                    component_number = int(c_number)
        return component_number+1

    @staticmethod
    def _get_component_name(component):
        return component.__class__.__name__
