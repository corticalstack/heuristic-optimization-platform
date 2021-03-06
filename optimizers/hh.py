from optimizers.hyper import Hyper
import logging
from utilities import logger as lg


class HH(Hyper):
    def __init__(self, **kwargs):
        Hyper.__init__(self, **kwargs)

    def optimize(self):
        self.hj.decay = self.decay  # Restore decay start-point to original configuration
        self.hyper()

    def hyper(self):
        self.set_llh_samples()
        self.add_samples_to_trend()
        bcf, bc, llh = self.select_heuristic()
        self.set_rbest(bcf, bc)

        while self.hj.budget > 0:
            bcf, bc, llh = self.select_heuristic()
            lg.msg(logging.DEBUG, 'Low level component {} seeding Hyper with best fitness {} and candidate {}'.format(
                self.low_level_heuristics[llh].oid, bcf, bc))
            self.set_rbest(bcf, bc)

            pop = self.set_pop()

            # Execute low level heuristic
            self.low_level_heuristics[llh].budget = self.hj.llh_budget
            self.low_level_heuristics[llh].rbest.fitness = self.hj.rbest.fitness
            self.low_level_heuristics[llh].rbest.candidate = self.hj.rbest.candidate
            self.low_level_heuristics[llh].population = pop

            self.low_level_heuristics[llh].oid_cls.run(fromhyper=True)
            self.low_level_heuristics[llh].llh_oid_run_count += 1

            self.hj.budget = int(self.hj.budget - self.hj.llh_budget)

            if self.low_level_heuristics[llh].rbest.fitness < self.hj.rbest.fitness:
                lg.msg(logging.INFO, 'Inserting fitness into archive {} by heuristic {}'.format(
                    self.low_level_heuristics[llh].rbest.fitness, self.low_level_heuristics[llh].oid))
                self.low_level_heuristics[llh].llh_oid_aggr_imp += (self.hj.rbest.fitness - self.low_level_heuristics[llh].rbest.fitness)
                self.llh_fitness[llh].insert(0, self.low_level_heuristics[llh].rbest.fitness)  # Insert at start
                self.llh_candidates[llh].insert(0, self.low_level_heuristics[llh].rbest.candidate)
                self.hj.rft.append(self.low_level_heuristics[llh].rbest.fitness)
                self.set_rbest(self.low_level_heuristics[llh].rbest.fitness, self.low_level_heuristics[llh].rbest.candidate)
                self.hj.iter_last_imp[self.hj.run] = self.hj.budget_total - self.hj.budget
                self.hj.imp_count[self.hj.run] += 1

    def select_heuristic(self):
        bcf, bc, llh = self.best_candidate_from_pool()

        if self.llh_total > 1 and self.hj.decay > self.random.random():
            choice = [i for i in range(0, self.llh_total) if i != llh]  # Exclude best
            llh = self.random.choice(choice)
            random_llh_best = min((v, c) for c, v in enumerate(self.llh_fitness[llh]))
            bcf = random_llh_best[0]
            bc = self.llh_candidates[llh][random_llh_best[1]]

        self.hj.decay *= self.hj.decay_coeff
        return bcf, bc, llh
