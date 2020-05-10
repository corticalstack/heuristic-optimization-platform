from controller import Controller
import logging
from utilities import logger as lg
from utilities.visualisation import Visualisation
import os
import sys
from datetime import datetime


class HeuristicOptimizerPlatform:
    """
    Heuristic Optimizer Platform
    """
    def __init__(self):
        lg.msg(logging.INFO, 'Heuristic Optimizer Platform (HOP) starting...')

        self.con = Controller()
        self.vis = Visualisation()
        self.optimize()
        lg.msg(logging.INFO, 'Heuristic Optimizer Platform (HOP) completed')

    def optimize(self):
        self.con.execute_jobs()


if __name__ == "__main__":
    log_filename = str('hop_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)8s] %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    hop = HeuristicOptimizerPlatform()

