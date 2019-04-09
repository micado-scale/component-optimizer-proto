class Config(object):

    def __init__(self, config, config_type):
        self._config = config.get(config_type)

    def get_property(self, property_name):
        return self._config.get(property_name)


class OptimizerConfig(Config):

    @property
    def nn_filename(self):
        return self.get_property('nn_filename')

    @nn_filename.setter
    def nn_filename(self, nn_filename):
        self._nn_filename = nn_filename

    @property
    def lr_filename(self):
        return self.get_property('lr_filename')

    @lr_filename.setter
    def lr_filename(self, lr_filename):
        self._lr_filename = lr_filename

    @property
    def constants_filename(self):
        return self.get_property('constants_filename')

    @constants_filename.setter
    def constants_filename(self, constants_filename):
        self._constants_filename = constants_filename

    @property
    def training_data_filename(self):
        return self.get_property('training_data_filename')

    @training_data_filename.setter
    def training_data_filename(self, training_data_filename):
        self._training_data_filename = training_data_filename
