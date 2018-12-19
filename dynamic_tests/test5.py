import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NUMBA_WARNINGS'] = '0'

import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython
from sklearn import svm
from icra_tools import statistics, utils, learner, knet
from icra_tools.expert import load_policy
from icra_tools.supervisor import NetSupervisor
from options import Options
import argparse
import scipy.stats
import pandas as pd

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--arch', required=True, nargs='+', type=int)
    ap.add_argument('--lr', required=False, type=int, default=.01)
    ap.add_argument('--epochs', required=False, type=int, default=100)
    ap.add_argument('--iters', required=True, type=int)
    ap.add_argument('--trials', required=True, type=int)
    ap.add_argument('--env', required=True)
    ap.add_argument('--t', required=True, type=int)

    # args = vars(ap.parse_args())
    opt = Options()

    # opt.envname = args['env']
    # opt.trials = args['trials']
    # opt.iters = args['iters']
    # opt.epochs = args['epochs']
    # opt.lr = args['lr']
    # opt.arch = args['arch']
    # opt.t = args['t']
    opt.load_args(ap.parse_args())

    opt.envname = opt.env
    opt.filename = '/Users/JonathanLee/experts/' + opt.envname + '.pkl'
    opt.env = gym.envs.make(opt.envname).env
    opt.sim = gym.envs.make(opt.envname).env
    opt.pi = load_policy.load_policy(opt.filename)
    opt.sess = tf.Session()
    opt.sup = NetSupervisor(opt.pi, opt.sess)

    plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
    data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    opt.plot_dir = plot_dir
    opt.data_dir = data_dir

    sup_rewards = np.zeros((opt.trials, opt.iters))
    lnr_rewards = np.zeros((opt.trials, opt.iters))
    rob_rewards = np.zeros((opt.trials, opt.iters))

    train_err = np.zeros((opt.trials, opt.iters))
    valid_err = np.zeros((opt.trials, opt.iters))
    test_err = np.zeros((opt.trials, opt.iters))
    robust_err = np.zeros((opt.trials, opt.iters))

    freq = np.zeros((opt.trials, opt.iters))

    for t in range(opt.trials):
        results = run_trial(opt)
        sup_rewards[t, :] = results['sup_reward']
        lnr_rewards[t, :] = results['lnr_reward']
        rob_rewards[t, :] = results['rob_reward']

        train_err[t, :] = results['train_err']
        valid_err[t, :] = results['valid_err']
        test_err[t, :] = results['test_err']
        robust_err[t, :] = results['robust_err']

        freq[t, :] = results['correction_freq']


    pd.DataFrame(sup_rewards).to_csv(opt.data_dir + 'sup_rewards.csv', index=False)
    pd.DataFrame(lnr_rewards).to_csv(opt.data_dir + 'lnr_rewards.csv', index=False)
    pd.DataFrame(rob_rewards).to_csv(opt.data_dir + 'rob_rewards.csv', index=False)

    pd.DataFrame(train_err).to_csv(opt.data_dir + 'train_err.csv', index=False)
    pd.DataFrame(valid_err).to_csv(opt.data_dir + 'valid_err.csv', index=False)
    pd.DataFrame(test_err).to_csv(opt.data_dir + 'test_err.csv', index=False)
    pd.DataFrame(robust_err).to_csv(opt.data_dir + 'robust_err.csv', index=False)
    
    pd.DataFrame(freq).to_csv(opt.data_dir + 'freq.csv', index=False)

    utils.plot([sup_rewards, lnr_rewards, rob_rewards], ['Supervisor', 'Learner', 'Robust Learner'], opt, "Reward", colors=['red', 'blue', 'green'])
    utils.plot([train_err, valid_err, test_err, robust_err], ['Training', 'Validation', 'Learner', 'Robust Learner'], opt, "Error", colors=['red', 'orange', 'blue', 'green'])
    utils.plot([freq], ['Frequency'], opt, 'Correction Frequency', colors=['green'])



def eval_oc(oc, X):
    preds = oc.predict(X)
    err = len(preds[preds == -1]) / float(len(preds))
    return err


def run_trial(opt):
    oc = svm.OneClassSVM(kernel='rbf', nu=.01, gamma = .01)
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)

    opt.samples = 5


    train_err = np.zeros(opt.iters)
    valid_err = np.zeros(opt.iters)
    test_err = np.zeros(opt.iters)
    robust_err = np.zeros(opt.iters)

    sup_reward = np.zeros(opt.iters)
    lnr_reward = np.zeros(opt.iters)
    rob_reward = np.zeros(opt.iters)

    correction_freq = np.zeros(opt.iters)

    for i in range(opt.iters):
        print "\nIteration: " + str(i)
        states, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t)

        lnr.add_data(states, int_actions)
        oc.fit(lnr.X)

        lnr.train()

        X_valid = []
        X_test = []
        X_robust = []
        sup_iter_rewards = np.zeros(opt.samples)
        lnr_iter_rewards = np.zeros(opt.samples)
        rob_iter_rewards = np.zeros(opt.samples)
        freqs = np.zeros(opt.samples)

        for j in range(opt.samples):
            states_valid, int_actions_valid, _, r_valid = statistics.collect_traj(opt.env, opt.sup, opt.t, False)
            states_test, int_actions_test, _, r_test, freq, lnr_score = statistics.collect_score_traj(opt.env, lnr, oc, opt.t, False)
            states_robust, int_actions_robust, _, r_robust, freq, rob_score = statistics.collect_robust_traj(opt.env, lnr, oc, opt.t, False)

            X_valid += states_valid
            X_test += states_test
            X_robust += states_robust

            sup_iter_rewards[j] = r_valid
            lnr_iter_rewards[j] = r_test
            rob_iter_rewards[j] = r_robust

            freqs[j] = freq

            if j == 0:
                utils.plot([np.array([lnr_score])], ['Learner'], opt, "DecisionScore" + str(i), colors=['blue'])
                utils.plot([np.array([rob_score])], ['Robust Learner'], opt, "RobustDecisionScore" + str(i), colors=['green'])


        train_err[i] = eval_oc(oc, lnr.X)
        valid_err[i] = eval_oc(oc, X_valid)
        test_err[i] = eval_oc(oc, X_test)
        robust_err[i] = eval_oc(oc, X_robust)

        sup_reward[i] = np.mean(sup_iter_rewards)
        lnr_reward[i] = np.mean(lnr_iter_rewards)
        rob_reward[i] = np.mean(rob_iter_rewards)

        correction_freq[i] = np.mean(freqs)

        print "One class train error: " + str(train_err[i])
        print "One class valid error: " + str(valid_err[i])

    return {
        "sup_reward": sup_reward,
        "lnr_reward": lnr_reward,
        "rob_reward": rob_reward,

        "train_err": train_err,
        "valid_err": valid_err,
        "test_err": test_err,
        "robust_err": robust_err,

        "correction_freq": correction_freq
    }
    



if __name__ == '__main__':
    main()




