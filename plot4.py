import numpy as np
import pickle
import matplotlib.pyplot as plt
import IPython
import argparse
from options import Options
from icra_tools import utils

ap = argparse.ArgumentParser()

ap.add_argument('--arch', required=True, nargs='+', type=int)
ap.add_argument('--lr', required=False, type=float, default=.01)
ap.add_argument('--epochs', required=False, type=int, default=100)
ap.add_argument('--iters', required=True, type=int)
ap.add_argument('--trials', required=True, type=int)
ap.add_argument('--env', required=True)
ap.add_argument('--t', required=True, type=int)
ap.add_argument('--grads', required=True, type=int)

ap.add_argument('--weights', required=True, nargs='+', type=float, default=[1.0, .1, .5])
ap.add_argument('--ufact', required=True, default=4.0, type=float)
ap.add_argument('--id', required=True, default=4.0, type=int)

ap.add_argument('--nu', required=True, type=float)
ap.add_argument('--gamma', required=True, type=float)

opt = Options()
args = ap.parse_args()
opt.load_args(args)
opt.envname = opt.env
args = vars(args)

plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))


trials_data = pickle.load(open(data_dir + 'opt/trials_data.pkl', 'r'))

for j, trial in enumerate(trials_data):
    rand_scores = trial['rand_scores']
    rand_cutoffs = trial['rand_cutoffs']
    approx_grad_scores = trial['approx_grad_scores']
    approx_grad_cutoffs = trial['approx_grad_cutoffs']
    finite_diff_scores = trial['finite_diff_scores']
    finite_diff_cutoffs = trial['finite_diff_cutoffs']

    els = 200

    plt.style.use('ggplot')
    for i, (rand_score, rand_cutoff) in enumerate(zip(rand_scores, rand_cutoffs)):
        plt.plot(rand_score[:els], color='blue', alpha=.5, label='Rand' if not i else '_nolegend')
        plt.plot(rand_cutoff[:els], color='black', linestyle='dashed', alpha=.5, label='Rand Cutoff' if not i else '_nolegend_')
    
    for i, (approx_grad_score, approx_grad_cutoff) in enumerate(zip(approx_grad_scores, approx_grad_cutoffs)):
        plt.plot(approx_grad_score[:els], color='red', alpha=.5, label='Approx Grad (AG)' if not i else '_nolegend_')
        plt.plot(approx_grad_cutoff[:els], color='black', linestyle='dotted', alpha=.5, label='AG Cutoff' if not i else '_nolegend_')

    for i, (finite_diff_score, finite_diff_cutoff) in enumerate(zip(finite_diff_scores, finite_diff_cutoffs)):
        plt.plot(finite_diff_score[:els], color='green', alpha=.5, label='Finite Diff (FD)' if not i else '_nolegend')
        plt.plot(finite_diff_cutoff[:els], color='black', alpha=.5, label='FD Cutoff' if not i else '_nolegned_')


    plt.legend()
    plt.show()
