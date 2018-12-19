from icra_tools import knet, net
import pickle
from icra_tools import pos_statistics as statistics
from icra_tools import utils
from icra_tools.supervisor import Supervisor
import IPython
from icra_tools import learner
import gym
import tensorflow as tf
import numpy as np
from sklearn.svm import OneClassSVM
import argparse
from options import Options
from icra_tools import learner
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

ap = argparse.ArgumentParser()

ap.add_argument('--weights', required=True, nargs='+', type=float, default=[1.0, .1, .5])
ap.add_argument('--ufact', required=True, default=4.0, type=float)
ap.add_argument('--id', required=True, default=4.0, type=int)
args = vars(ap.parse_args())

envname = 'Pusher6-v0'
exp_id = args['id']
env = gym.envs.make(envname).env
env.my_weights = args['weights']
env.ufact = args['ufact']
net = net.Network([64, 64], .01, 300)
#net.load_weights('meta/weights.txt', 'meta/stats.txt')
suffix = '_' + utils.stringify(args['weights']) + '_' + str(args['ufact'])
weights_path = 'meta/' + 'test' + '/' + envname + '_' + str(exp_id) + '_weights' + suffix + '.txt'
stats_path = 'meta/' + 'test' + '/' + envname + '_' + str(exp_id) + '_stats' + suffix + '.txt'
net.load_weights(weights_path, stats_path) 

net_sup = Supervisor(net)

opt = Options
opt.env = env
opt.sup = net_sup
opt.t = 100

est = knet.Network([64, 64], learning_rate=.01, epochs=100)
lnr = learner.Learner(est)

oc = OneClassSVM(kernel='rbf', gamma = .01, nu = .01)

ITERATIONS = 500
print "\n\nSup rollouts\n\n"
sup_failures = 0
initial_states = []
for i in range(ITERATIONS):
    print "iteration: " + str(i)
    violation = True
    while violation:
        states, int_actions, taken_actions, r, violation = statistics.collect_traj_rejection(opt.env, opt.sup, opt.t, False, False)
        if violation:
            print "Violation at iteration " + str(i) + ", restarting"
            sup_failures += 1


    initial_states.append(states[0])
    lnr.add_data(states, int_actions)


print "Sup failures: " + str(sup_failures / float(ITERATIONS))

# oc.fit(initial_states)


# print "\n\nValidation rollouts\n\n"

# pred_y = []
# actual_y = []
# valid_violations = 0
# for i in range(ITERATIONS):
#     results = statistics.collect_traj_rejection(opt.env, opt.sup, opt.t, True, False)
#     states = results[0]
#     pred_safe = oc.predict([states[0]])[0] == 1
#     safe = not results[-1]

#     pred_y.append(pred_safe)
#     actual_y.append(safe)

#     if results[-1]:
#         valid_violations += 1

# pred_y = np.array(pred_y).astype(int)
# actual_y = np.array(actual_y).astype(int)
# rand_y = np.random.randint(0, 2, len(actual_y))

# valid_recall = recall_score(actual_y, pred_y)
# valid_precision = precision_score(actual_y, pred_y)
# rand_valid_recall = recall_score(actual_y, rand_y)
# rand_valid_precision = precision_score(actual_y, rand_y)

# print "\n\nLearner rollouts\n\n"

# lnr.train()
# pred_y = []
# actual_y = []
# test_violations = 0
# for i in range(ITERATIONS):
#     results = statistics.collect_traj_rejection(opt.env, lnr, opt.t, True, False)
#     states = results[0]
#     pred_safe = oc.predict([states[0]])[0] == 1
#     safe = not results[-1]
#     print "iteration: " + str(i) + ", pred " + str(pred_safe)
#     print "iteration: " + str(i) + ", " + str(safe)
#     print "matching: " + str(safe == pred_safe) + "\n"

#     pred_y.append(pred_safe)
#     actual_y.append(safe)

#     if results[-1]:
#         test_violations += 1

# pred_y = np.array(pred_y).astype(int)
# actual_y = np.array(actual_y).astype(int)
# rand_y = np.random.randint(0, 2, len(actual_y))

# test_recall = recall_score(actual_y, pred_y)
# test_precision = precision_score(actual_y, pred_y)
# rand_test_recall = recall_score(actual_y, rand_y)
# rand_test_precision = precision_score(actual_y, rand_y)


# print "\n\n\n\n"
# print "RESULTS\n"
# print "Validation recall: " + str(valid_recall)
# print "Validation precision: " + str(valid_precision)
# print "\n"
# print "Test recall: " + str(test_recall)
# print "Test precision: " + str(test_precision)
# print "\n\n\n\n"

# print "Random validation recall: " + str(rand_valid_recall)
# print "Random validation precision: " + str(rand_valid_precision)
# print "Random test recall: " + str(rand_test_recall)
# print "Random test precision: " + str(rand_test_precision)

# print "\n\n"
# print "Valid violations: " + str(float(valid_violations) / ITERATIONS)
# print "Test violations: "  +str(float(test_violations) / ITERATIONS)
# print "\n\n"

# print "\n\n\n\n"
# IPython.embed()






