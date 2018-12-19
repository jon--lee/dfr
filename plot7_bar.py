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


all_trials = pickle.load(open(data_dir + 'multiple_trials/trials_data.pkl', 'r'))
print "Averaging over " + str(len(all_trials)) + " trials"

van_tallies = []
rand_tallies = []
es_tallies = []

for trials_data in all_trials:
    # print "\nTrials: " + str(len(trials_data))

    for trial in trials_data:
        van_tallies.append(trial['van_tallies'])
        rand_tallies.append(trial['rand_tallies'])
        es_tallies.append(trial['es_tallies'])


samples = 1.0
van_tallies = np.array(van_tallies) / samples
rand_tallies = np.array(rand_tallies) / samples
es_tallies = np.array(es_tallies) / samples

van_tallies[:, -1] = 1 - van_tallies[:, 0] - van_tallies[:, 1]
rand_tallies[:, -1] = 1 - rand_tallies[:, 0] - rand_tallies[:, 1]
es_tallies[:, -1] = 1 - es_tallies[:, 0] - es_tallies[:, 1]

van_means, van_sems = statistics.mean_sem(van_tallies)
rand_means, rand_sems = statistics.mean_sem(rand_tallies)
es_means, es_sems = statistics.mean_sem(es_tallies)

plt.style.use('ggplot')
width = .2
index = np.arange(len(van_means))

def plot(x, y, color, label=None):
    if not label:
        plt.plot(x, y, solid_capstyle="butt", linewidth=25, color=color)
    else:
        plt.plot(x, y, solid_capstyle="butt", linewidth=2, color=color, label=label)
    return 

def make_points(means):
    points = []
    last = 0
    for i, value in enumerate(means):
        points.append([last, last + value])
        last = last + value

    return points

means = [van_means, es_means, rand_means]
sems = [van_sems, es_sems, rand_sems]

line1 = np.ones(2) * -1
line2 = np.ones(2) * 0
line3 = np.ones(2) * 1

print "van means:  " + str(van_means)
print "es means:   " + str(es_means)
print "rand means: " + str(rand_means)

plt.figure(figsize=(7, 2.5))
van_points = make_points(van_means)
labels=['Completed', 'Failed', 'Failed Safely']
colors = ['#368BBD', '#E24B33', '#998FC5']
for point, color, label in zip(van_points, colors, labels):
    plot(point, line3, color)

es_points = make_points(es_means)
for point, color in zip(es_points, colors):
    plot(point, line2, color)

for point, color, label in zip(van_points, colors, labels):
    plot(point, line3, color, label)


rand_points = make_points(rand_means)
for point, color in zip(rand_points, colors):
    plot(point, line1, color)

plt.xlim(0, 1)

yticks = ['DFFA', 'ES', 'Baseline']
y = [-1, 0, 1]
plt.ylim(-1.5, 1.5)
plt.yticks(y, yticks)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
plt.savefig('tmp.svg')









# IPython.embed()

labels = ['Naive', 'Early Stopping', 'Recovery']
tabs = ["Completed", "Failed", "Incomplete"]
for i, (mean, sem, label) in enumerate(zip(means, sems, labels)[:3]):
    plt.bar(index + (i - 1) * width, mean, width, label=label, yerr=sem)

#print means
print "Completed comp: " + str( rand_means[0] / van_means[0] )
print "Failed comp:    " + str(rand_means[1] / van_means[1])


plt.legend()
plt.xticks(index, tabs)
plt.ylim(0, 1)
plt.savefig("tmp_plots/" + opt.envname + "gamma" + str(opt.gamma) + "_nu" + str(opt.nu) + ".svg")
# plt.show()

