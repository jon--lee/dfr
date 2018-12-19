import numpy as np
import pickle
import matplotlib.pyplot as plt
import IPython
import argparse
from options import Options
from icra_tools import utils
from icra_tools import statistics

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


trials_data = pickle.load(open(data_dir + 'trials_data.pkl', 'r'))
print "\nTrials: " + str(len(trials_data))

rand_scores_aggregate = []
ag_scores_aggregate = []
fd_scores_aggregate = []
els = 200


van_tallies = np.zeros((len(trials_data), 3))
rand_tallies = np.zeros((len(trials_data), 3))
es_tallies = np.zeros((len(trials_data), 3))
ag_tallies = np.zeros((len(trials_data), 3))
fd_tallies = np.zeros((len(trials_data), 3))

for j, trial in enumerate(trials_data):
    van_tallies[j, :] = trial['van_tallies']
    rand_tallies[j, :] = trial['rand_tallies']
    es_tallies[j, :] = trial['es_tallies']
    ag_tallies[j, :] = trial['ag_tallies']
    fd_tallies[j, :] = trial['fd_tallies']

samples = 5.0
van_tallies = van_tallies / samples
van_tallies[:, 2] = 1.0 - van_tallies[:, 0] - van_tallies[:, 1]
rand_tallies = rand_tallies / samples
rand_tallies[:, 2] = 1.0 - rand_tallies[:, 0] - rand_tallies[:, 1]
es_tallies = es_tallies / samples
es_tallies[:, 2] = 1.0 - es_tallies[:, 0] - es_tallies[:, 1]
ag_tallies = ag_tallies / samples
ag_tallies[:, 2] = 1.0 - ag_tallies[:, 0] - ag_tallies[:, 1]
fd_tallies = fd_tallies / samples
fd_tallies[:, 2] = 1.0 - fd_tallies[:, 0] - fd_tallies[:, 1]



van_means, van_sems = statistics.mean_sem(van_tallies)
rand_means, rand_sems = statistics.mean_sem(rand_tallies)
es_means, es_sems = statistics.mean_sem(es_tallies)
ag_means, ag_sems = statistics.mean_sem(ag_tallies)
fd_means, fd_sems = statistics.mean_sem(fd_tallies)

print "Van:      " + str(np.mean(van_tallies, axis=0))
print "Early:    " + str(np.mean(es_tallies, axis=0))
print "Rand:     " + str(np.mean(rand_tallies, axis=0))
print "AG:     " + str(np.mean(ag_tallies, axis=0))
print "FD:     " + str(np.mean(fd_tallies, axis=0))

print np.sum(fd_means), np.sum(ag_means), np.sum(es_means), np.sum(van_means), np.sum(rand_means)

plt.style.use('ggplot')
width = .2
index = np.arange(len(van_means))

means = [van_means, es_means, rand_means, ag_means, fd_means]
sems = [van_sems, es_sems, rand_sems, ag_sems, fd_sems]
labels = ['Naive', 'Early Stopping', 'Recovery', "Recovery (AG)", "Recovery (FD)"]
tabs = ["Completed", "Failed", "Incomplete"]
for i, (mean, sem, label) in enumerate(zip(means, sems, labels)[:3]):
    plt.bar(index + (i - 1) * width, mean, width, label=label, yerr=sem)

plt.legend()
plt.xticks(index, tabs)
plt.ylim(0, 1)
plt.savefig("pusher_bars.svg")
plt.show()




