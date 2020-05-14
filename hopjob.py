from optimizers.particle import Particle


class HopJob:
    def __init__(self):
        self.step = 0
        self.pid_enabled = False
        self.oid_enabled = False
        self.type = None
        self.low_level_selection_pool = []
        self.llh_sample_runs = 0
        self.llh_sample_budget = 0
        self.llh_budget = 0
        self.pid = None
        self.pid_cls = None
        self.pid_lb = 0
        self.pid_ub = 0
        self.oid = None
        self.pid_lb_diff_pct = 0
        self.pid_ub_diff_pct = 0
        self.oid_cls = None
        self.oid_lb = 0
        self.oid_ub = 0
        self.oid_run_count = 0
        self.oid_aggr_imp = 0
        self.bid = None
        self.comp_budget_base = 0
        self.budget = 0
        self.runs_per_optimizer = 0
        self.bit_computing = 0
        self.initial_sample = False
        self.initial_candidate_size = 0
        self.number_parents = 0
        self.number_children = 0
        self.sample_size_coeff = 0.01  # Usually used as n dim * (budget * sample size coeff)
        self.generator = None
        self.variator = None
        self.rbest = Particle()
        self.gbest = Particle()
        self.rft = []
        self.gft = []
        self.population = []
        self.start_time = 0
        self.end_time = 0
        self.total_comp_time_s = 0
        self.avg_comp_time_s = 0
        self.inertia_coeff = 0.0
        self.local_coeff = 0.0
        self.global_coeff = 0.0
        self.decay = 0
        self.decay_coeff = 0.0




