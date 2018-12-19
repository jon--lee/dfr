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



keys = sorted(all_trials[0].keys())

all_van_means = []
all_van_sems = []

all_rand_means = []
all_rand_sems = []

all_es_means = []
all_es_sems = []

for key in keys:

    van_tallies = []
    rand_tallies = []
    es_tallies = []

    for trials_data in all_trials:
        van_samples = []
        rand_samples = []
        es_samples = []

        for trial in trials_data[key]:
            van_samples.append(trial['van_tallies'])
            rand_samples.append(trial['rand_tallies'])
            es_samples.append(trial['es_tallies'])


        van_tallies.append(np.mean(van_samples, axis=0))
        rand_tallies.append(np.mean(rand_samples, axis=0))
        es_tallies.append(np.mean(es_samples, axis=0))


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

    # van_means, van_sems = van_means[::-1], van_sems[::-1]
    # rand_means, rand_sems = rand_means[::-1], rand_sems[::-1]
    # es_means, es_sems = es_means[::-1], es_sems[::-1]

    all_van_means.append(van_means)
    all_van_sems.append(van_sems)

    all_rand_means.append(rand_means)
    all_rand_sems.append(rand_sems)

    all_es_means.append(es_means)
    all_es_sems.append(es_sems)


def switch(data):
    return data

all_van_means = switch(np.array(all_van_means))
all_van_sems = switch(np.array(all_van_sems))
all_rand_means = switch(np.array(all_rand_means))
all_rand_sems = switch(np.array(all_rand_sems))
all_es_means = switch(np.array(all_es_means))
all_es_sems = switch(np.array(all_es_sems))





plt.style.use('ggplot')
tabs = ["Completed", "Collided", "Halted"]

# colors = ["#CB0017", "#89D77D", "#5486C0"]
colors = ["#FC8540", "#89D77D", "#2A76AF"]

co = 100

# f, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
# for j, ax in enumerate(axes):
#     if j == 1:
#         j = 2
#     elif j == 2:
#         j = 1

#     p = ax.plot(keys[:co], all_van_means[:co, j], label='Baseline', color=colors[0])
#     ax.fill_between(keys[:co], -all_van_sems[:co, j] + all_van_means[:co, j], all_van_sems[:co, j] + all_van_means[:co, j], alpha=.5, color=p[0].get_color())

#     p = ax.plot(keys[:co][:1], all_van_means[:co, j][:1])
#     p = ax.plot(keys[:co][:1], all_van_means[:co, j][:1])
#     p = ax.plot(keys[:co], all_es_means[:co, j], label='ES')
#     ax.fill_between(keys[:co], -all_es_sems[:co, j] + all_es_means[:co, j], all_es_sems[:co, j] + all_es_means[:co, j], alpha=.5, color=p[0].get_color())

#     p = ax.plot(keys[:co], all_rand_means[:co, j], label='DFR', color=colors[2])
#     ax.fill_between(keys[:co], -all_rand_sems[:co, j] + all_rand_means[:co, j], all_rand_sems[:co, j] + all_rand_means[:co, j], alpha=.5, color=p[0].get_color())




#     ax.set_xlabel("Demonstrations", fontsize=14)
#     ax.set_title(tabs[j], fontsize=16)

# plt.tight_layout()


van_comps = all_van_means[:co, 0]
van_halts = all_van_means[:co, 2]
van_colls = all_van_means[:co, 1]

es_comps = all_es_means[:co, 0]
es_halts = all_es_means[:co, 2]
es_colls = all_es_means[:co, 1]

rand_comps = all_rand_means[:co, 0]
rand_halts = all_rand_means[:co, 2]
rand_colls = all_rand_means[:co, 1]


weights = [1.0, 0.0, -1.0]
van_res = weights[0] * van_comps + weights[1] * van_halts + weights[2] * van_colls
es_res = weights[0] * es_comps + weights[1] * es_halts + weights[2] * es_colls
rand_res = weights[0] * rand_comps + weights[1] * rand_halts + weights[2] * rand_colls

plt.tight_layout()
plt.xlabel('Demonstrations', fontsize=14)

van_p = plt.plot(keys[:co], van_res, label='Baseline', color=colors[0])
# plt.plot(es_res[:1])
plt.plot(es_res[:1])
es_p = plt.plot(keys[:co], es_res, label='ES')
rand_p = plt.plot(keys[:co], rand_comps, label='DFR')


plt.legend()
plt.ylim(-.1, 1.1)

# plt.show()
plt.savefig('tmp_plots/weight_-5.0.pdf')

plt.close()
plt.cla()
plt.clf()

van_res = np.mean(van_res)
es_res = np.mean(es_res)
rand_res = np.mean(rand_res)

print(van_res, es_res, rand_res)

ind = np.arange(3)
plt.xticks(ind, ('Baseline', 'ES', 'DFR'))
barlist = plt.bar(ind, [van_res, es_res, rand_res])
barlist[0].set_color(van_p[0].get_color())
barlist[1].set_color(es_p[0].get_color())
barlist[2].set_color(rand_p[0].get_color())

plt.savefig('tmp_plots/weights.pdf')





# width = .2
# index = np.arange(len(van_means))


# means = [van_means, es_means, rand_means]
# sems = [van_sems, es_sems, rand_sems]

# # IPython.embed()

# labels = ['Baseline', 'Early Stopping', 'Failure-Avoidance']
# tabs = ["Completed", "Failed", "Incomplete"][::-1]
# for i, (mean, sem, label) in enumerate(zip(means, sems, labels)[:3]):
#     plt.bar(index + (i - 1) * width, mean, width, label=label, yerr=sem)

# #print means
# print "Completed comp: " + str( rand_means[0] / van_means[0] )
# print "Failed comp:    " + str(rand_means[1] / van_means[1])


# plt.legend()
# plt.xticks(index, tabs)
# plt.ylim(0, 1)
# plt.savefig("tmp_plots/" + opt.envname + "gamma" + str(opt.gamma) + "_nu" + str(opt.nu) + ".svg")
# # plt.show()

