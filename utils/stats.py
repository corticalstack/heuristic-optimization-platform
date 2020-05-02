from scipy.stats import ranksums
import logging
from utils import logger as lg
import statistics
from collections import OrderedDict


class Stats:
    @staticmethod
    def summary(optimizers):
        lg.msg(logging.INFO, 'Basic Statistics')
        reference_sample = []
        comparison_sample = []
        alpha = 0.05
        summary_results = OrderedDict()

        oi = 0  # Counter for enabled optimizers, with first as ref
        for opt in optimizers:
            if not optimizers[opt]['enabled']:
                continue
            stdev = round(statistics.pstdev(optimizers[opt]['ft']), 3)
            mean = round(statistics.mean(optimizers[opt]['ft']), 3)
            minf = min(optimizers[opt]['ft'])
            maxf = max(optimizers[opt]['ft'])
            wts = ' '  # Wilcoxon test symbol
            if oi == 0:
                reference_sample = optimizers[opt]['ft']
                ref_mean = mean
            elif oi > 0:  # Wilcoxon is pairwise comparison so makes no sense without at least 1 pair
                comparison_sample = optimizers[opt]['ft']
                zstat, pvalue = ranksums(reference_sample, comparison_sample)
                wts = '='
                if pvalue < alpha:
                    if ref_mean > mean:
                        wts = '-'
                    else:
                        wts = '+'
            summary_results[opt] = {'minf': minf, 'maxf': maxf, 'mean': mean, 'stdev': stdev, 'wts': wts, 'avg_cts':
                optimizers[opt]['avg_cts'], 'lb_diff_pct': optimizers[opt]['lb_diff_pct'], 'ub_diff_pct':
                optimizers[opt]['ub_diff_pct']}
            oi += 1

        lg.msg(logging.INFO, 'Optimiser\tMin Fitness\tMax Fitness\tAvg Fitness\tStDev\tWilcoxon\tLB Diff %\tUB Diff %\tAvg Cts')
        for k, v in summary_results.items():
            lg.msg(logging.INFO, '{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}\t\t{}\t\t{}\t\t{}'.format(
                str(k), str(v['minf']), str(v['maxf']), str(v['mean']), str(v['stdev']), str(v['wts']),
                str(v['lb_diff_pct']), str(v['ub_diff_pct']), str(round(v['avg_cts'], 3))))

    @staticmethod
    def taillard_compare(lb, ub, alg_fitness):
        return round(((alg_fitness - lb) / lb) * 100, 2), round(((alg_fitness - ub) / ub) * 100, 2)

