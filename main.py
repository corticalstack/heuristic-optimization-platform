from heuristics_manager import HeuristicsManager
import logging
from utilities import logger as lg
from utilities.visualisation import Visualisation
import os
from datetime import datetime


class HeuristicOptimizerPlatform:
    """
    Heuristic Optimizer Platform
    """
    def __init__(self):
        self.results_path = 'results/hoprun_' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.create_results_folder()
        self.set_log_file()
        lg.msg(logging.INFO, 'Heuristic Optimizer Platform (HOP) starting...')

        self.hm = HeuristicsManager(results_path=self.results_path)
        self.vis = Visualisation()
        self.optimize()
        lg.msg(logging.INFO, 'Heuristic Optimizer Platform (HOP) completed')

    def create_results_folder(self):
        try:
            os.mkdir(self.results_path)
        except OSError:
            print('Creation of results directory {} failed'.format(self.results_path))
        else:
            print('Successfully created results directory {}'.format(self.results_path))

    def set_log_file(self):
        log_filename = str(self.results_path + '/' + 'hoplog_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='[%(asctime)s] [%(levelname)8s] %(message)s')

        # Disable matplotlib font manager logger
        logging.getLogger('matplotlib.font_manager').disabled = True

    def optimize(self):
        self.hm.execute_jobs()


if __name__ == "__main__":
    hop = HeuristicOptimizerPlatform()

