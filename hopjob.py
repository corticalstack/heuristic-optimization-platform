from optimizers.particle import Particle


class HopJob:
    def __init__(self):
        self.step = 0

        # Problem, optimizer, benchmark and type
        self.pid = None
        self.oid = None
        self.bid = None
        self.type = None

        # Active component flags
        self.pid_enabled = False
        self.oid_enabled = False

        # Active component classes
        self.pid_cls = None
        self.oid_cls = None

        # Low Level Heuristics
        self.low_level_selection_pool = []
        self.llh_sample_runs = 0
        self.llh_sample_budget = 0
        self.llh_budget = 0

        # Computational Budget
        self.runs_per_optimizer = 0
        self.comp_budget_base = 0
        self.budget = 0

        # Binary Encoding
        self.bit_computing = 16

        # Sampling
        self.initial_sample = False

        # Bounds
        self.pid_lb = 0
        self.pid_ub = 0
        self.pid_lb_diff_pct = 0
        self.pid_ub_diff_pct = 0
        self.oid_lb = 0
        self.oid_ub = 0

        # Population
        self.population = []
        self.number_parents = 0
        self.number_children = 0
        self.initial_pop_size = 0

        # Solution generator and variator
        self.generator_comb = None
        self.generator_cont = None
        self.variator = None

        # Runtime stats
        self.rbest = Particle()
        self.gbest = Particle()
        self.rft = []
        self.gft = []
        self.llh_oid_run_count = 0
        self.llh_oid_aggr_imp = 0
        self.start_time = 0
        self.end_time = 0
        self.total_comp_time_s = 0
        self.avg_comp_time_s = 0

        # Various co-efficients
        self.sample_size_coeff = 0.01  # Usually used as n dim * (budget * sample size coeff)
        self.inertia_coeff = 0.0
        self.local_coeff = 0.0
        self.global_coeff = 0.0
        self.decay = 0
        self.decay_coeff = 0.0
