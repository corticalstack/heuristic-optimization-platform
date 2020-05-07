from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg


class Hyper(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)
        # Persist current configuration and problem

        self.range = 1000  # +/- value the randomly select number can be between
        self.bounds = 2000  # Action space bounds

        self.action_space = [0, 1]
        #self.observation_space = spaces.Discrete(5)

        self.number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        pass
        #self.np_random, seed = seeding.np_random(seed)
        #return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        if action < self.number:
            self.observation = 1

        elif action == self.number:
            self.observation = 2

        elif action > self.number:
            self.observation = 3

        # JP could set reward according to whether fitness value beats best

        reward = ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}

    def reset(self):
        #self.number = self.np_random.uniform(-self.range, self.range)
        #self.guess_count = 0
        #self.observation = 0
        return self.observation
