"""
This one is working fine for experiments
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBA_WARNINGS'] = '0'

from icra_tools import knet, net
import pickle
from icra_tools import pos_statistics as statistics
from icra_tools import rec_statistics
from icra_tools import utils
from icra_tools.supervisor import Supervisor
import IPython
from icra_tools import learner
import gym
import tensorflow as tf
import numpy as np
# from sklearn import svm
from icra_tools.ocsvm import OCSVM
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
    args = vars(args)

    opt.envname = opt.env
    opt.env = gym.envs.make(opt.envname).env
    opt.sim = gym.envs.make(opt.envname).env
    exp_id = args['id']
    opt.env.my_weights = args['weights']
    opt.env.ufact = args['ufact']
    opt.pi = net.Network([64, 64], .01, 150)
    suffix = '_' + utils.stringify(args['weights']) + '_' + str(args['ufact'])
    weights_path = 'meta/' + 'test' + '/' + opt.envname + '_' + str(exp_id) + '_weights' + suffix + '.txt'
    stats_path = 'meta/' + 'test' + '/' + opt.envname + '_' + str(exp_id) + '_stats' + suffix + '.txt'
    opt.pi.load_weights(weights_path, stats_path) 
    opt.sup = Supervisor(opt.pi)
    opt.misc = Options()
    opt.misc.num_evaluations = 1


    opt.misc.samples = 5
    rec_results = {}
    lnr_results = {}

    plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
    data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir  + '/multiple_trials'):
        os.makedirs(data_dir + '/multiple_trials')
    if not os.path.exists(plot_dir + '/multiple_trials'):
        os.makedirs(plot_dir + '/multiple_trials')

    opt.data_dir = data_dir
    opt.plot_dir = plot_dir

    opt.env.reset()
    opt.env.render()

    trials_data = []
    for t in range(opt.trials):
        print "\n\nTrial: " + str(t) + "\n\n"
        opt.t_value = t
        trials_data.append(run_trial(opt))

        print "Saving to: "  + str(opt.data_dir + 'multiple_trials/trials_data.pkl')
        # pickle.dump(trials_data, open(opt.data_dir + 'multiple_trials/trials_data.pkl', 'w'))

    


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

    if info['first_out'] > -1 and info['first_violation'] > -1 and info['first_out'] <= info['first_violation']:
        print "correctly identified out of distr"
        true_positive = 1

    if info['first_out'] == -1 and info['first_violation'] == -1:
        print "all good"
        true_negative = 1

    return false_negative, false_positive, true_positive, true_negative

def check_completed(info):
    complete_t = info['first_complete']
    violation_t = info['first_violation']

    results = {
        'comp_before_fail': 0,
        'comp_before_alarm': 0
    }

    if info['first_complete'] > -1 and (info['first_violation'] == -1 or info['first_violation'] > info['first_complete']):
        print "Completed before a failure"
        results['comp_before_fail'] = 1

    if info['first_complete'] > -1 and info['first_violation'] == -1 and (info['first_violation'] > info['first_complete'] or info['first_violation'] == -1):
        print "Completed before alarm"
        results['comp_before_alarm'] = 1

    return results


def run_trial(opt):
    ocs = [ OCSVM(kernel='rbf', gamma = opt.gamma, nu = opt.nu) for t in range(opt.t) ]
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
            # lnr.train()
            print "\t\tFitting oc svms..."
            # fit_all(ocs, trajs_train)
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

    pickle.dump(lnr.X, open('data/lnrX.pkl', 'w'))
    pickle.dump(lnr.y, open('data/lnry.pkl', 'w'))
    pickle.dump(trajs_train, open('data/trajs_train.pkl', 'w'))
    pickle.dump(actions_train, open('data/actions_train.pkl', 'w'))


    # print "Loading data..."
    # lnr.X = pickle.load(open('data/lnrX.pkl', 'r'))
    # lnr.y = pickle.load(open('data/lnry.pkl', 'r'))
    # lnr.X = lnr.X
    # lnr.y = lnr.y
    # trajs_train = pickle.load(open('data/trajs_train.pkl', 'r'))
    # actions_train = pickle.load(open('data/actions_train.pkl', 'r'))
    # print "Done loading data."


    trajs = trajs_train
    fit_all(ocs, trajs)


    print "Training net..."
    lnr.train()
    print "Fitting svms..."
    # trajs_train = trajs[:-200]
    # trajs_test = trajs[-200:]
    # fit_all(ocs, trajs_train)
    # print eval_ocs(ocs, trajs_train)
    # print eval_ocs(ocs, trajs_test)
    print "Done fitting"

    Ls = np.zeros((len(trajs_train), opt.t))
    KLs = np.zeros((len(trajs_train), opt.t))
    state_diffs = np.zeros((len(trajs_train), opt.t))
    func_diffs = np.zeros((len(trajs_train), opt.t))
    action_norms = np.zeros((len(trajs_train), opt.t))
    actions = np.zeros((len(trajs_train), opt.t, opt.env.action_space.shape[0]))

    for i, (traj_states, traj_actions) in enumerate(zip(trajs_train, actions_train)):
        zipped = zip(traj_states, traj_actions)
        for t, (state, action) in enumerate(zipped[:-1]):
            state_next, action_next = zipped[t+1]
            state_diff = np.linalg.norm(state_next - state)
            func_diff = np.abs(ocs[t].decision_function([state])[0,0] - ocs[t].decision_function([state_next])[0,0])
            action_norm = np.linalg.norm(action)
            
            Ls[i, t] = state_diff / action_norm
            KLs[i, t] = func_diff / action_norm
            state_diffs[i, t] = state_diff
            func_diffs[i, t] = func_diff
            action_norms[i, t] = action_norm
            actions[i, t, :] = action

    max_Ls = np.amax(Ls, axis=0)
    max_KLs = np.amax(KLs, axis=0)

    max_rec = 500

    trial_data = []
    for k in range(30):
        reject = True
        while reject:
            print "Determing whether to reject initial state..."
            s = opt.env.reset()
            reject = ocs[0].predict([s])[0] == -1
        init_state = opt.env.get_pos_vel()

        print "\n\nVanilla Learner\n\n"

        van_completed = 0
        van_failed = 0
        van_failed_in_support = 0

        for i in range(opt.misc.samples):
            print "Eval Iteration: " + str(i)
            results = rec_statistics.collect_traj(opt.env, lnr, ocs, opt.t, visualize=True, early_stop=False, init_state = init_state)
            van_completed += int(results[-3]['completed'])
            van_failed += int(results[-3]['failed'])
            van_failed_in_support += int(results[-3]['failed_in_support'])

        van_completed = van_completed / float(opt.misc.samples)
        van_failed = van_failed / float(opt.misc.samples)
        van_failed_in_support = van_failed_in_support / float(opt.misc.samples)

        print "\n\nRand Control Recovery Strategy\n\n"

        rand_completed = 0
        rand_failed = 0
        rand_failed_in_support = 0

        for i in range(opt.misc.samples):
            print "Eval Iteration: " + str(i)
            results = rec_statistics.collect_rec(rec_statistics.random_sample_loop, opt.env, opt.sim, lnr, ocs, opt.t, opt, max_KLs, visualize=True, early_stop=False, init_state=init_state, max_rec=max_rec)
            rand_completed += int(results[-3]['completed'])
            rand_failed += int(results[-3]['failed'])
            rand_failed_in_support += int(results[-3]['failed_in_support'])

        if rand_failed > 0:
            IPython.embed()

        rand_completed = rand_completed / float(opt.misc.samples)
        rand_failed = rand_failed / float(opt.misc.samples)
        rand_failed_in_support = rand_failed_in_support / float(opt.misc.samples)

        print "\n\nEarly Stopping Strategy\n\n"
        es_completed = 0
        es_failed = 0
        es_failed_in_support = 0

        for i in range(opt.misc.samples):
            print "Eval Iteration: " + str(i)
            results = rec_statistics.collect_rec(rec_statistics.no_rec_loop, opt.env, opt.sim, lnr, ocs, opt.t, opt, max_KLs, visualize=True, early_stop=False, init_state=init_state, max_rec=max_rec)
            es_completed += int(results[-3]['completed'])
            es_failed += int(results[-3]['failed'])
            es_failed_in_support += int(results[-3]['failed_in_support'])

        es_completed = es_completed / float(opt.misc.samples)
        es_failed = es_failed / float(opt.misc.samples)
        es_failed_in_support = es_failed_in_support / float(opt.misc.samples)

        results = {
            'van_tallies': [van_completed, van_failed, van_failed_in_support],
            'rand_tallies': [rand_completed, rand_failed, rand_failed_in_support],
            'es_tallies': [es_completed, es_failed, es_failed_in_support],
        }

        trial_data.append(results)
        # print "Saving to: " + opt.data_dir + 'multiple_trials/trial_data' + str(opt.t_value) + '.pkl'
        # pickle.dump(trial_data, open(opt.data_dir + 'multiple_trials/trial_data' + str(opt.t_value) + '.pkl', 'w'))


    return trial_data



if __name__ == '__main__':
    main()





