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

print "\n"
print "nu:    " + str(args['nu'])
print "gamma: " + str(args['gamma'])


plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))


trials_data = pickle.load(open(data_dir + 'opt/trials_data.pkl', 'r'))

rand_scores_aggregate = []
ag_scores_aggregate = []
fd_scores_aggregate = []
els = 200


rand_tallies = np.zeros((len(trials_data), 3))
ag_tallies = np.zeros((len(trials_data), 3))
fd_tallies = np.zeros((len(trials_data), 3))
for j, trial in enumerate(trials_data):
    for rand_score, rand_cutoff in zip(trial['rand_scores'], trial['rand_cutoffs']):
        rand_scores_aggregate.append((rand_score/rand_cutoff)[:els])

    for ag_score, ag_cutoff in zip(trial['approx_grad_scores'], trial['approx_grad_cutoffs']):
        ag_scores_aggregate.append((ag_score/ag_cutoff)[:els])

    for fd_score, fd_cutoff in zip(trial['finite_diff_scores'], trial['finite_diff_cutoffs']):
        fd_scores_aggregate.append((fd_score/fd_cutoff)[:els])

    rand_tallies[j, :] = np.array(trial['rand_tallies']) / 10.0
    ag_tallies[j, :] = trial['approx_grad_tallies']
    fd_tallies[j, :] = trial['finite_diff_tallies']


print "Rand:  " + str(np.mean(rand_tallies, axis=0))
print "AG:    " + str(np.mean(ag_tallies, axis=0))
print "FD:    " + str(np.mean(fd_tallies, axis=0))

plt.style.use('ggplot')

rand_scores_aggregate = np.array(rand_scores_aggregate)
ag_scores_aggregate = np.array(ag_scores_aggregate)
fd_scores_aggregate = np.array(fd_scores_aggregate)

# rand_means = np.mean(rand_scores_aggregate, axis=0)
# ag_means = np.mean(ag_scores_aggregate, axis=0)
# fd_means = np.mean(fd_scores_aggregate, axis=0)

# rand_sems = scipy.stats.sem(rand_scores_aggregate, axis=0)
# ag_sems = scipy.stats.sem(ag_scores_aggregate, axis=0)
# fd_sems = scipy.stats.sem(fd_scores_aggregate, axis=0)

datas = [rand_scores_aggregate, ag_scores_aggregate, fd_scores_aggregate]
labels = ['Rand', 'Approx Grad', 'Finite Diff']
colors = ['blue', 'red', 'green']
utils.plot(datas, labels, opt, "Optimize", colors)

# plt.plot(rand_scores_aggregate[:els], color='blue', alpha=.5, label='Rand' )
# plt.plot(ag_scores_aggregate[:els], color='red', alpha=.5, label='Rand' )
# plt.plot(fd_scores_aggregate[:els], color='green', alpha=.5, label='Rand')

# for j, trial in enumerate(trials_data):
#     rand_scores = trial['rand_scores']
#     rand_cutoffs = trial['rand_cutoffs']
#     approx_grad_scores = trial['approx_grad_scores']
#     approx_grad_cutoffs = trial['approx_grad_cutoffs']
#     finite_diff_scores = trial['finite_diff_scores']
#     finite_diff_cutoffs = trial['finite_diff_cutoffs']

#     els = 200

#     plt.style.use('ggplot')
#     for i, (rand_score, rand_cutoff) in enumerate(zip(rand_scores, rand_cutoffs)):
#         plt.plot(rand_score[:els], color='blue', alpha=.5, label='Rand' if not i else '_nolegend')
#         plt.plot(rand_cutoff[:els], color='black', linestyle='dashed', alpha=.5, label='Rand Cutoff' if not i else '_nolegend_')
    
#     for i, (approx_grad_score, approx_grad_cutoff) in enumerate(zip(approx_grad_scores, approx_grad_cutoffs)):
#         plt.plot(approx_grad_score[:els], color='red', alpha=.5, label='Approx Grad (AG)' if not i else '_nolegend_')
#         plt.plot(approx_grad_cutoff[:els], color='black', linestyle='dotted', alpha=.5, label='AG Cutoff' if not i else '_nolegend_')

#     for i, (finite_diff_score, finite_diff_cutoff) in enumerate(zip(finite_diff_scores, finite_diff_cutoffs)):
#         plt.plot(finite_diff_score[:els], color='green', alpha=.5, label='Finite Diff (FD)' if not i else '_nolegend')
#         plt.plot(finite_diff_cutoff[:els], color='black', alpha=.5, label='FD Cutoff' if not i else '_nolegned_')


#     plt.legend()
#     plt.show()
