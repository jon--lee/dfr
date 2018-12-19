import numpy as np
import pickle
import matplotlib.pyplot as plt
import IPython

trials_data = pickle.load(open('results/trials_data_oracle.pkl', 'r'))

for j, trial in enumerate(trials_data):
    rand_scores = trial['rand_scores']
    rand_cutoffs = trial['rand_cutoffs']
    approx_grad_scores = trial['approx_grad_scores']
    approx_grad_cutoffs = trial['approx_grad_cutoffs']
    finite_diff_scores = trial['finite_diff_scores']
    finite_diff_cutoffs = trial['finite_diff_cutoffs']

    els = 200

    plt.style.use('ggplot')
    for i, (rand_score, rand_cutoff, approx_grad_score, approx_grad_cutoff, finite_diff_score, finite_diff_cutoff) in enumerate(zip(rand_scores, rand_cutoffs, approx_grad_scores, approx_grad_cutoffs, finite_diff_scores, finite_diff_cutoffs)):
        plt.plot(rand_score[:els], color='blue', alpha=.5, label='Rand' if not i else '_nolegend_')
        plt.plot(approx_grad_score[:els], color='red', alpha=.5, label='Approx Grad (AG)' if not i else '_nolegend_')
        plt.plot(finite_diff_score[:els], color='green', alpha=.5, label='Finite Diff (FD)' if not i else '_nolegend')

        plt.plot(rand_cutoff[:els], color='black', linestyle='dashed', alpha=.5, label='Rand Cutoff' if not i else '_nolegend_')
        plt.plot(approx_grad_cutoff[:els], color='black', linestyle='dotted', alpha=.5, label='AG Cutoff' if not i else '_nolegend_')
        plt.plot(finite_diff_cutoff[:els], color='black', alpha=.5, label='FD Cutoff' if not i else '_nolegned_')
    plt.legend()
    plt.show()
