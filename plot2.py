import numpy as np
import pickle
import matplotlib.pyplot as plt
import IPython

trials_data = pickle.load(open('results/trials_data.pkl', 'r'))

for j, trial in enumerate(trials_data):
    rand_scores = trial['rand_scores']
    rand_cutoffs = trial['rand_cutoffs']
    approx_grad_scores = trial['approx_grad_scores']
    approx_grad_cutoffs = trial['approx_grad_cutoffs']

    els = 200

    plt.style.use('ggplot')
    for i, (rand_score, rand_cutoff, approx_grad_score, approx_grad_cutoff) in enumerate(zip(rand_scores, rand_cutoffs, approx_grad_scores, approx_grad_cutoffs)):
        plt.plot(rand_score[:els]/rand_cutoff[:els], color='blue', alpha=.6, label='Rand' if not i else '_nolegend_')
        plt.plot(approx_grad_score[:els]/approx_grad_cutoff[:els], color='red', alpha=.6, label='Approx Grad (AG)' if not i else '_nolegend_')

    plt.legend()
    plt.show()
