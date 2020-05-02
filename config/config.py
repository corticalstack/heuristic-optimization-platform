import yaml


class Config:
    def __init__(self):
        self.settings = {}
        self.files = {
            'gen': 'config/general.yaml',
            'prb': 'config/problems.yaml',
            'opt': 'config/optimizers.yaml',
            'ben': 'config/benchmarks.yaml'
        }

        self.get_config()

    def get_config(self):
        for k, v, in self.files.items():
            with open(v, 'r') as stream:
                try:
                    self.settings[k] = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print(e)
