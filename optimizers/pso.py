from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import copy
from operator import attrgetter


class PSO(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.gbest_swarm = []
        self.prev_swarm = []

        self.velocity_min = -self.hj.oid_ub
        self.velocity_max = self.hj.oid_ub

    def optimize(self):
        self.swarm()

    def swarm(self):
        # Incoming population migrates to starting population, reset continuous permutation values
        if self.hj.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        for i in range(self.hj.initial_pop_size - len(self.hj.population)):
            new_c = Particle()
            
            # Generate candidate of cont values within domain bounds
            new_c.candidate_cont = self.get_generator()(lb=self.hj.oid_lb, ub=self.hj.oid_ub)

            if self.hj.pid_type == 'combinatorial':
                # Transform candidate of cont values back to discrete using smallest position value method
                new_c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(new_c.candidate_cont)
            else:
                new_c.candidate = new_c.candidate_cont

            new_c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(new_c.candidate, self.hj.budget)
            
            self.hj.population.append(new_c)

        self.gbest_swarm = copy.deepcopy(self.hj.population)
        self.prev_swarm = copy.deepcopy(self.hj.population)
        self.set_rbest(min(self.hj.population, key=attrgetter('fitness')))

        while self.hj.budget > 0:
            new_swarm = self.swarm_in_motion()
            self.prev_swarm = copy.deepcopy(self.hj.population)
            self.hj.population = copy.deepcopy(new_swarm)

            # Evaluate population fitness and set personal (local) best
            for ci, c in enumerate(self.hj.population):
                c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate, self.hj.budget)
                if c.fitness < self.gbest_swarm[ci].fitness:
                    self.gbest_swarm[ci] = copy.deepcopy(c)
                if c.fitness < self.hj.rbest.fitness:
                    self.set_rbest(c)
                    self.hj.rft.append(c.fitness)
                    if not self.fromhyper:
                        self.hj.iter_last_imp[self.hj.run] = self.hj.budget_total - self.hj.budget
                        self.hj.imp_count[self.hj.run] += 1

    def set_rbest(self, candidate):
        self.hj.rbest = copy.deepcopy(candidate)

    def swarm_in_motion(self):
        new_s = []
        for ci, c in enumerate(self.hj.population):
            new_c = Particle()  # New candidate particle
            new_c.candidate_cont = []
            for pi, p in enumerate(c.candidate_cont):
                exp_inertia = p + self.hj.inertia_coeff * (p - self.prev_swarm[ci].candidate_cont[pi])
                exp_local = self.hj.local_coeff * self.random.random() * (self.gbest_swarm[ci].candidate_cont[pi] - p)
                exp_global = self.hj.global_coeff * self.random.random() * (self.hj.rbest.candidate_cont[pi] - p)
                velocity = exp_inertia + exp_local + exp_global
                new_c.candidate_cont.append(velocity)

            if self.hj.pid_type == 'combinatorial':
                new_c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(new_c.candidate_cont)
            else:
                new_c.candidate_cont = self.clamp(new_c.candidate_cont)
                new_c.candidate = new_c.candidate_cont
            new_s.append(new_c)

        return new_s

    def clamp(self, candidate):
        new_candidate = []
        for c in candidate:
            new_candidate.append(max(min(self.hj.pid_ub, c), self.hj.pid_lb))
        return new_candidate

    def reset_inherited_population_attr(self):
        for c in self.hj.population:
            c.candidate_cont = self.hj.pid_cls.candidate_spv_discrete_to_continuous(c.candidate, self.hj.pid_lb,
                                                                                    self.hj.pid_ub)
