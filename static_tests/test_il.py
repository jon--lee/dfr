import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBA_WARNINGS'] = '0'

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
from sklearn import svm
import argparse
from options import Options
from icra_tools import learner
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import time as timer
import matplotlib.pyplot as plt
import scipy.stats


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--arch', required=True, nargs='+', type=int)
    ap.add_argument('--lr', required=False, type=int, default=.01)
    ap.add_argument('--epochs', required=False, type=int, default=100)
    ap.add_argument('--iters', required=True, type=int)
    ap.add_argument('--trials', required=True, type=int)
    ap.add_argument('--env', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--grads', required=True, type=int)

    ap.add_argument('--weights', required=True, nargs='+', type=float, default=[1.0, .1, .5])
    ap.add_argument('--ufact', required=True, default=4.0, type=float)
    ap.add_argument('--id', required=True, default=4.0, type=int)

    opt = Options()
    args = ap.parse_args()
    opt.load_args(args)
    args = vars(args)

    opt.envname = opt.env
    opt.env = gym.envs.make(opt.envname).env
    exp_id = args['id']
    opt.env.my_weights = args['weights']
    opt.env.ufact = args['ufact']
    opt.pi = net.Network([64, 64], .01, 300)
    suffix = '_' + utils.stringify(args['weights']) + '_' + str(args['ufact'])
    weights_path = 'meta/' + 'test' + '/' + opt.envname + '_' + str(exp_id) + '_weights' + suffix + '.txt'
    stats_path = 'meta/' + 'test' + '/' + opt.envname + '_' + str(exp_id) + '_stats' + suffix + '.txt'
    opt.pi.load_weights(weights_path, stats_path) 
    opt.sup = Supervisor(opt.pi)
    opt.misc = Options()
    opt.misc.num_evaluations = 1


    opt.misc.samples = 600
    rec_results = {}
    lnr_results = {}

    try:
        for t in range(opt.trials):
            start_time = timer.time()
            results = run_trial(opt)

            for key in results['rec'].keys():
                if key in rec_results:
                    rec_results[key].append(results['rec'][key])
                else:
                    rec_results[key] = [results['rec'][key]]

            for key in results['lnr'].keys():
                if key in lnr_results:
                    lnr_results[key].append(results['lnr'][key])
                else:
                    lnr_results[key] = [results['lnr'][key]]

    except KeyboardInterrupt:
        pass

    labels = sorted(list(rec_results.keys()))
    for key in rec_results.keys():
        rec_results[key] = np.array(rec_results[key]) / float(opt.misc.samples)
        lnr_results[key] = np.array(lnr_results[key]) / float(opt.misc.samples)

    rec_means = [np.mean(rec_results[key]) for key in sorted(rec_results.keys())]
    rec_sems = [scipy.stats.sem(rec_results[key]) for key in sorted(rec_results.keys())]
    lnr_means = [np.mean(lnr_results[key]) for key in sorted(lnr_results.keys())]
    lnr_sems = [scipy.stats.sem(lnr_results[key]) for key in sorted(lnr_results.keys())]
    
    plt.style.use('ggplot')
    width = .4
    index = np.arange(len(rec_means))

    plt.bar(index + 0 * width, rec_means, width, label='recovery')
    plt.bar(index + 1 * width, lnr_means, width, label='no recovery')
    plt.legend()
    plt.xticks(index, labels)
    plt.ylim(0, 1)
    plt.show()




def eval_oc(oc, X):
    preds = oc.predict(X)
    err = len(preds[preds == -1]) / float(len(preds))
    return err


def fit_all(ocs, trajs):
    for t, oc in enumerate(ocs):
        X_train = []
        for traj in trajs:
            X_train.append(traj[t])
        oc.fit(X_train)

def eval_ocs(ocs, trajs):
    T = len(ocs)
    errs = np.zeros(T)
    trajs_array = np.array(trajs)
    try:
        for t in range(T):
            X = trajs_array[:, t, :]
            errs[t] = eval_oc(ocs[t], X)
    except IndexError:
        print "Index error, do something?"
        IPython.embed()
    return errs

def check_predictions(info):
    false_negative = 0
    false_positive = 0
    true_positive = 0
    true_negative = 0

    if info['first_violation'] > -1 and (info['first_out'] == -1 or info['first_out'] > info['first_violation']):
        print "\tdidn't pick up on failure..."
        false_negative = 1

    if info['first_out'] > -1 and (info['first_violation'] == -1):
        print "\tOverly conservative..."
        false_positive = 1

    if info['first_out'] > -1 and info['first_violation'] > -1 and info['first_out'] < info['first_violation']:
        print "correctly identified out of distr"
        true_positive = 1

    if info['first_out'] == -1 and info['first_violation'] == -1:
        print "all good"
        true_negative = 1

    return false_negative, false_positive, true_positive, true_negative




def run_trial(opt):
    ocs = [ svm.OneClassSVM(kernel='rbf', gamma = .05, nu = .05) for t in range(opt.t) ]
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)
    opt.samples = 1

    sup_reward = np.zeros(opt.misc.num_evaluations)
    lnr_reward = np.zeros(opt.misc.num_evaluations)
    rob_reward = np.zeros(opt.misc.num_evaluations)

    train_err = np.zeros((opt.misc.num_evaluations, opt.t))
    valid_err = np.zeros((opt.misc.num_evaluations, opt.t))
    test_err = np.zeros((opt.misc.num_evaluations, opt.t))
    robust_err = np.zeros((opt.misc.num_evaluations, opt.t))
    correction_freq = np.zeros(opt.misc.num_evaluations)

    trajs_train = []
    actions_train = []
    for i in range(opt.iters):

        print "Iteration: " + str(i)
        violation = True
        while violation:
            states, int_actions, taken_actions, r, violation = statistics.collect_traj_rejection(opt.env, opt.sup, opt.t, False, False)
            if violation:
                print "\tViolation, restarting"

        trajs_train.append(states)
        actions_train.append(int_actions)
        lnr.add_data(states, int_actions)


        if (i + 1) % (opt.iters/opt.misc.num_evaluations) == 0:
            print "\tEvaluating..."
            print "\t\tTraining learner..."
            lnr.train()
            print "\t\tFitting oc svms..."
            fit_all(ocs, trajs_train)
            print "\t\tDone fitting"

            trajs_valid = []
            trajs_test = []
            trajs_robust = []

            sup_iters_rewards = np.zeros(opt.samples)
            lnr_iters_rewards = np.zeros(opt.samples)
            rob_iters_rewards = np.zeros(opt.samples)
            freqs = np.zeros(opt.samples)

            # for j in range(opt.samples):
                # states_valid, int_actions_valid, _, r_valid = statistics.collect_traj(opt.env, opt.sup, opt.t, False, False)
                # states_test, int_actions_test, _, r_test, _, lnr_score, violation = statistics.collect_score_traj_multiple_rejection(opt.env, lnr, ocs, opt.t, False, False)
                # states_robust, int_actions_robust, _, r_robust, freq, rob_score, mags = statistics.collect_robust_traj_multiple(opt.env, lnr, ocs, opt.t, opt, False, False)

                # trajs_valid.append(states_valid)
                # trajs_test.append(states_test)
                # trajs_robust.append(states_robust)

                # sup_iters_rewards[j] = r_valid
                # lnr_iters_rewards[j] = r_test
                # rob_iters_rewards[j] = r_robust

                # freqs[j] = freq

                # if j == 0:
                #     utils.plot([np.array([lnr_score]), np.array([rob_score])], ['Learner', 'Robust Learner'], opt, "scores/DecisionScores" + str(i), colors=['blue', 'green'])
                #     utils.plot([np.array([mags])], ['Robust Learner'], opt, "mags/RecoveryMagnitudes" + str(i), colors=['green'])


            index = i / (opt.iters / opt.misc.num_evaluations)
            # train_err[index, :] = eval_ocs(ocs, trajs_train)
            # valid_err[index, :] = eval_ocs(ocs, trajs_valid)
            # test_err[index, :] = eval_ocs(ocs, trajs_test)
            # robust_err[index, :] = eval_ocs(ocs, trajs_robust)

            # sup_reward[index] = np.mean(sup_iters_rewards)
            # lnr_reward[index] = np.mean(lnr_iters_rewards)
            # rob_reward[index] = np.mean(rob_iters_rewards)

            # correction_freq[index] = np.mean(freqs)

    # pickle.dump(lnr.X, open('data/lnrX.pkl', 'w'))
    # pickle.dump(lnr.y, open('data/lnry.pkl', 'w'))
    # pickle.dump(trajs_train, open('data/trajs_train.pkl', 'w'))
    # pickle.dump(actions_train, open('data/actions_train.pkl', 'w'))

    # print "Loading data..."
    # lnr.X = pickle.load(open('data/lnrX.pkl', 'r'))
    # lnr.y = pickle.load(open('data/lnry.pkl', 'r'))
    # trajs_train = pickle.load(open('data/trajs_train.pkl', 'r'))
    # actions_train = pickle.load(open('data/actions_train.pkl', 'r'))
    # print "Done loading data."


    print "Training net..."
    lnr.train()
    print "Fitting svms..."
    fit_all(ocs, trajs_train)
    print "Done fitting"

    # Ks = []
    # ts = []
    # Ls = []
    # max_K = 0
    # max_L = 0
    # max_Kt = -1
    # max_Lt = -1
    # for traj_states, traj_actions in zip(trajs_train, actions_train):
    #     zipped = zip(traj_states, traj_actions)
    #     for t, (state, action) in enumerate(zipped[:-1]):
    #         state_next, action_next = zipped[t+1]
    #         state_norm = np.linalg.norm(state_next - state)
    #         func_diff = np.abs(ocs[t].decision_function([state])[0,0] - ocs[t].decision_function([state_next])[0,0])
    #         action_norm = np.linalg.norm(action)
            
    #         K = state_norm / action_norm
    #         L = func_diff / action_norm

    #         Ls.append(L)
    #         Ks.append(K)
    #         ts.append(t)
    #         if K > max_K:
    #             max_K = K
    #             max_Kt = t
    #         if L > max_L:
    #             max_L = L
    #             max_Lt = t

    # print "Max train L: " + str(max_L)
    # print "Avg train L: " + str(np.mean(Ls))
    # print "Max train K: " + str(max_K)
    # print "Avg train K: " + str(np.mean(Ks))


    IPython.embed()

    # print "\n\nRecovery\n\n"


    # rec_failures = 0
    # false_negatives = 0
    # false_positives = 0
    # true_positives = 0
    # true_negatives = 0
    # rec_failed = 0


    # for j in range(opt.misc.samples):
    #     print "Iteration: " + str(j)
    #     reject = True
    #     while reject:
    #         results = statistics.collect_robust_traj_multiple_rejection_adaptive(opt.env, lnr, ocs, opt.t, opt, max_L, False, False)
    #         reject = results[-1]
    #         failed = results[-2]
    #         if reject:
    #             print "\tRejecting " + str(j) + " and restarting..."
    #     info = results[-3]

    #     tup = check_predictions(info)
    #     false_negatives += tup[0]
    #     false_positives += tup[1]
    #     true_positives += tup[2]
    #     true_negatives += tup[3]

    #     if failed:
    #         rec_failures += 1
    #         print "\t" + str(j) + " failed..."

    #     if info['rec_failed'] > -1:
    #         rec_failed += 1
    #         print "\t rec failed"


    # rec_results = {
    #     "failures": rec_failures,
    #     "false_negatives": false_negatives,
    #     "false_positives": false_positives,
    #     "true_positives": true_positives,
    #     "true_negatives": true_negatives,
    #     "caused_fail": rec_failed,
    # }
    # print "fraction of failures: " + str(rec_failures / float(opt.misc.samples))


    # print "\n\nLearner\n\n"

    # lnr_failures = 0
    # false_negatives = 0
    # false_positives = 0
    # true_positives = 0
    # true_negatives = 0

    # for j in range(opt.misc.samples):
    #     print "Iteration: " + str(j)
    #     reject = True
    #     while reject:
    #         results = statistics.collect_score_traj_multiple_rejection(opt.env, lnr, ocs, opt.t, False, False)
    #         reject = results[-1]
    #         failed = results[-2]
    #         info = results[-3]
    #         if reject:
    #             print "\tRejecting " + str(j) + " and restarting..."

    #     tup = check_predictions(info)
    #     false_negatives += tup[0]
    #     false_positives += tup[1]
    #     true_positives += tup[2]
    #     true_negatives += tup[3]

    #     if failed:
    #         lnr_failures += 1
    #         print "\t" + str(j) + " failed..."
    # print "fraction of failures: " + str(lnr_failures / float(opt.misc.samples))

    # lnr_results = {
    #     "failures": lnr_failures,
    #     "false_negatives": false_negatives,
    #     "false_positives": false_positives,
    #     "true_positives": true_positives,
    #     "true_negatives": true_negatives,
    #     "caused_fail": 0
    # }

    # print "\n\n\nrec_results"
    # print rec_results

    # print "lnr_results"
    # print lnr_results

    # return {
    #     "rec": rec_results,
    #     "lnr": lnr_results,
    # }
    return None


if __name__ == '__main__':
    main()





