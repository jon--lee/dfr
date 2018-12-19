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
from svm_tools.traj_sv import TrajSV

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--arch', required=True, nargs='+', type=int)
    ap.add_argument('--lr', required=False, type=int, default=.01)
    ap.add_argument('--epochs', required=False, type=int, default=100)
    ap.add_argument('--iters', required=True, type=int)
    ap.add_argument('--trials', required=True, type=int)
    ap.add_argument('--env', required=True)
    ap.add_argument('--t', required=True, type=int)

    opt = Options()
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

def eval_ocs(ocs, trajs, opt):
    errs = np.zeros(opt.t)
    trajs_array = np.array(trajs)
    for t in range(opt.t):
        X = trajs_array[:, t, :]
        errs[t] = eval_oc(ocs[t], X)
    return errs



def run_trial(opt):
    ocs = [ svm.OneClassSVM(kernel='rbf', gamma = .5, nu = .01) for t in range(opt.t) ]
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)
    opt.samples = 10

    sup_reward = np.zeros(opt.iters)
    lnr_reward = np.zeros(opt.iters)
    rob_reward = np.zeros(opt.iters)

    train_err = np.zeros(opt.iters)
    valid_err = np.zeros(opt.iters)
    test_err = np.zeros(opt.iters)
    robust_err = np.zeros(opt.iters)
    correction_freq = np.zeros(opt.iters)

    trajs_train = []

    for i in range(opt.iters):
        print "\nIteration: " + str(i)
        states, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t, False)
        trajs_train.append(states)

        lnr.add_data(states, int_actions)
        lnr.train()
        
        trajs_train_array = np.array(trajs_train)
        for t in range(opt.t):
            X = trajs_train_array[:, t, :]
            ocs[t].fit(X)

        if i % 5 == 0:
            trajs_valid = []
            for j in range(opt.samples):
                states_valid, int_actions_valid, _, r_valid = statistics.collect_traj(opt.env, opt.sup, opt.t, False)
                trajs_valid.append(states_valid)

            train_oc_errs = eval_ocs(ocs, trajs_train, opt)
            valid_oc_errs = eval_ocs(ocs, trajs_valid, opt)

            print "Train errs: " + str(train_oc_errs)
            print "Valid errs: " + str(valid_oc_errs)
            print "Max train err: " + str(np.amax(train_oc_errs))
            print "Max valid err: " + str(np.amax(valid_oc_errs))

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




