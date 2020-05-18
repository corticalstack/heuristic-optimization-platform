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
            if isinstance(c[0], float) and args['slf'].hj.pid_type == 'combinatorial':
                c = args['slf'].hj.pid_cls.candidate_spv_continuous_to_discrete(c)
            f, _ = args['slf'].hj.pid_cls.evaluator(c)
            fitness.append(f)
            args['slf'].hj.budget -= 1
        return fitness

    @staticmethod
    def observer(population, num_generations, num_evaluations, args):
        if args['slf'].hj.oid_cls.__class__.__name__ == 'DEA':
            ft = [o.fitness for o in population]
            args['slf'].hj.rft.extend(ft)
        else:
            # Persist best fitness as population evolves. Note use of max is correct irrespective of max or min problem,
            # as Inspyred knows which type of problem the heuristic is instantiated with
            best = max(population)
            args['slf'].hj.rft.append(round(best.fitness, 2))

        args['slf'].hj.rft.sort()
        if args['slf'].hj.rft[0] < args['slf'].hj.rbest.fitness:
            args['slf'].hj.rbest.fitness = args['slf'].hj.rft[0]
            if not args['slf'].fromhyper:
                args['slf'].hj.iter_last_imp[args['slf'].hj.run] = args['slf'].hj.budget_total - args['slf'].hj.budget
                args['slf'].hj.imp_count[args['slf'].hj.run] += 1
