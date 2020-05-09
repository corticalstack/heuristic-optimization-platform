from optimizers.hyper import Hyper
import copy

class HH(Hyper):
    def __init__(self, **kwargs):
        Hyper.__init__(self, **kwargs)

        self.decay = 0.5
        self.decay_factor = 0.97

    def optimize(self):
        self.decay = 0.6
        self.hyper()

    def hyper(self):
        #self.budget *= 20
        self.set_llh_samples()
        self.add_samples_to_trend()

        while self.budget > 0:
            bcf, bc, llh = self.select_heuristic()
            self.set_gbest(bcf, bc)

            pop = self.set_pop()
            cls = self.low_level_heuristics[llh]['cls']

            # Execute low level heuristic
            gbest = copy.deepcopy(self.gbest)
            run_best, run_ft, run_budget = cls.run(budget=self.llh_budget, gbest=gbest, pop=pop)
            self.low_level_heuristics[llh]['run_count'] += 1

            print('Run best is ', run_best.fitness, ' with candidate ', run_best.candidate, ' by ', self.low_level_heuristics[llh]['llh'])
            print('Global best is ', self.gbest.fitness)

            self.budget += run_budget
            self.budget -= self.llh_budget

            if run_best.fitness < self.gbest.fitness:
                print('Inserting fitness into archive ', run_best.fitness)
                self.low_level_heuristics[llh]['aggr_imp'] += (self.gbest.fitness - run_best.fitness)
                self.llh_fitness[llh].insert(0, run_best.fitness)  # Insert at start
                self.llh_candidates[llh].insert(0, run_best.candidate)
                self.fitness_trend.append(run_best.fitness)
                self.set_gbest(run_best.fitness, run_best.candidate)

    def select_heuristic(self):
        bcf, bc, llh = self.best_candidate_from_pool()

        if self.llh_total > 1 and self.random.random() < self.decay:
            choice = [i for i in range(0, self.llh_total) if i != llh]
            llh = self.random.choice(choice)

        self.decay *= self.decay_factor
        return bcf, bc, llh
