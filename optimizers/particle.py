class Particle:
    def __init__(self, fitness_default=999999999):
        self.fitness_default = fitness_default
        self.candidate = []
        self.fitness = fitness_default
