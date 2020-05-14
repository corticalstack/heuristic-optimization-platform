import copy


class InspyredWrapper:
    def __init__(self):
        pass

    @staticmethod
    def generator(random, args):
        if args['slf'].hj.population:
            candidate = copy.deepcopy(args['slf'].hj.population[0].candidate)
            args['slf'].hj.population.pop(0)
        else:
            candidate = args['slf'].get_generator()(lb=args['slf'].hj.pid_lb, ub=args['slf'].hj.pid_ub)
        return candidate

    @staticmethod
    def evaluator(candidates, args):
        fitness = []
        for c in candidates:
            if isinstance(c[0], float) and args['slf'].hj.type == 'combinatorial':
                c = args['slf'].hj.pid_cls.candidate_spv_continuous_to_discrete(c)
            f, _ = args['slf'].hj.pid_cls.evaluator(c)
            fitness.append(f)
            args['slf'].hj.budget -= 1
        return fitness

    @staticmethod
    def observer(population, num_generations, num_evaluations, args):
        if args['slf'].hj.oid_cls.__class__.__name__ == 'DEA':
            args['slf'].hj.rft = [o.fitness for o in population]
        else:
            best = max(population)  # Persist best fitness as population evolves
            args['slf'].hj.rft.append(round(best.fitness, 2))
