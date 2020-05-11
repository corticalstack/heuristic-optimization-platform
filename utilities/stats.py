from scipy.stats import ranksums
import statistics
from collections import OrderedDict


class Stats:
    @staticmethod
    def get_summary(trends):
        reference_sample = []
        comparison_sample = []
        alpha = 0.05
        summary_results = OrderedDict()

        oi = 0  # Counter for enabled optimizers, with first as ref
        for k, v in trends.items():
            stdev = round(statistics.pstdev(v), 3)
            mean = round(statistics.mean(v), 3)
            minf = min(v)
            maxf = max(v)
            wts = ' '  # Wilcoxon test symbol
            if oi == 0:
                reference_sample = v
                ref_mean = mean
            elif oi > 0:  # Wilcoxon is pairwise comparison so makes no sense without at least 1 pair
                comparison_sample = v
                zstat, pvalue = ranksums(reference_sample, comparison_sample)
                wts = '='
                if pvalue < alpha:
                    if ref_mean > mean:
                        wts = '-'
                    else:
                        wts = '+'
            summary_results[k] = {'minf': minf, 'maxf': maxf, 'mean': mean, 'stdev': stdev, 'wts': wts}
            oi += 1
        return summary_results

    @staticmethod
    def taillard_compare(lb, ub, alg_fitness):
        return round(((alg_fitness - lb) / lb) * 100, 2), round(((alg_fitness - ub) / ub) * 100, 2)

