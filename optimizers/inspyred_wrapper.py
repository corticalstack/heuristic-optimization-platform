class InspyredWrapper:
    def __init__(self):
        pass

    @staticmethod
    def generator(random, args):
        perm = getattr(args['prb'], 'generator_' + args['cfg'].settings['opt']['ES']['generator'])(args['slf'].pos_min,
                                                                                                   args['slf'].pos_max,)
        return perm

    @staticmethod
    def evaluator(candidates, args):
        fitness = []
        for c in candidates:
            if isinstance(c[0], float):
                c = args['slf'].prb.perm_spv_continuous_to_discrete(c)
            f = args['prb'].evaluator(c)
            fitness.append(f)
        return fitness

    @staticmethod
    def observer(population, num_generations, num_evaluations, args):
        if args['slf'].__class__.__name__ == 'DEA':
            args['slf'].fitness_trend = [o.fitness for o in population]
        else:
            best = max(population)  # Persist best fitness as population evolves
            args['slf'].fitness_trend.append(best.fitness)
