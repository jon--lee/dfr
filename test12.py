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
import time as timer

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
    opt.misc = Options()
    opt.misc.num_evaluations = 50 


    plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
    data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    opt.plot_dir = plot_dir
    opt.data_dir = data_dir

    sup_rewards = np.zeros((opt.trials, opt.misc.num_evaluations))
    lnr_rewards = np.zeros((opt.trials, opt.misc.num_evaluations))
    rob_rewards = np.zeros((opt.trials, opt.misc.num_evaluations))

    train_err = np.zeros((opt.trials, opt.misc.num_evaluations, opt.t))
    valid_err = np.zeros((opt.trials, opt.misc.num_evaluations, opt.t))
    test_err = np.zeros((opt.trials, opt.misc.num_evaluations, opt.t))
    robust_err = np.zeros((opt.trials, opt.misc.num_evaluations, opt.t))

    freq = np.zeros((opt.trials, opt.misc.num_evaluations))

    print "Running Trials:\n\n"

    try:
        for t in range(opt.trials):
            start_time = timer.time()
            results = run_trial(opt)
            sup_rewards[t, :] = results['sup_reward']
            lnr_rewards[t, :] = results['lnr_reward']
            rob_rewards[t, :] = results['rob_reward']

            train_err[t, :, :] = results['train_err']
            valid_err[t, :, :] = results['valid_err']
            test_err[t, :, :] = results['test_err']
            robust_err[t, :, :] = results['robust_err']

            freq[t, :] = results['correction_freq']

            sup_rewards_save, lnr_rewards_save, rob_rewards_save = sup_rewards[:t, :], lnr_rewards[:t+1, :], rob_rewards[:t+1, :]
            train_err_save, valid_err_save, test_err_save, robust_err_save = train_err[:t+1, :, :], valid_err[:t+1, :, :], test_err[:t+1, :, :], robust_err[:t+1, :, :]
            freq_save = freq[:t+1, :]

            pd.DataFrame(sup_rewards_save).to_csv(opt.data_dir + 'sup_rewards.csv', index=False)
            pd.DataFrame(lnr_rewards_save).to_csv(opt.data_dir + 'lnr_rewards.csv', index=False)
            pd.DataFrame(rob_rewards_save).to_csv(opt.data_dir + 'rob_rewards.csv', index=False)

            for tau in range(opt.t):
                pd.DataFrame(train_err_save[:, :, tau]).to_csv(opt.data_dir + 'train_err_t' + str(tau) + '.csv', index=False)
                pd.DataFrame(valid_err_save[:, :, tau]).to_csv(opt.data_dir + 'valid_err_t' + str(tau) + '.csv', index=False)
                pd.DataFrame(test_err_save[:, :, tau]).to_csv(opt.data_dir + 'test_err_t' + str(tau) + '.csv', index=False)
                pd.DataFrame(robust_err_save[:, :, tau]).to_csv(opt.data_dir + 'robust_err_t' + str(tau) + '.csv', index=False)
            
            pd.DataFrame(freq_save).to_csv(opt.data_dir + 'freq.csv', index=False)

            train_err_avg = np.mean(train_err_save, axis=2)
            valid_err_avg = np.mean(valid_err_save, axis=2)
            test_err_avg = np.mean(test_err_save, axis=2)
            robust_err_avg = np.mean(robust_err_save, axis=2)

            utils.plot([sup_rewards_save, lnr_rewards_save, rob_rewards_save], ['Supervisor', 'Learner', 'Robust Learner'], opt, "Reward", colors=['red', 'blue', 'green'])
            utils.plot([train_err_avg, valid_err_avg, test_err_avg, robust_err_avg], ['Training', 'Validation', 'Learner', 'Robust Learner'], opt, "Error", colors=['red', 'orange', 'blue', 'green'])
            utils.plot([freq_save], ['Frequency'], opt, 'Correction Frequency', colors=['green'])
            
            end_time = timer.time()
            print "Total time: " + str(end_time - start_time)

    except KeyboardInterrupt:
        pass

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
    for t in range(T):
        X = trajs_array[:, t, :]
        errs[t] = eval_oc(ocs[t], X)
    return errs


def run_trial(opt):
    ocs = [ svm.OneClassSVM(kernel='rbf', gamma = .05, nu = .05) for t in range(opt.t) ]
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)
    opt.samples = 100

    sup_reward = np.zeros(opt.misc.num_evaluations)
    lnr_reward = np.zeros(opt.misc.num_evaluations)
    rob_reward = np.zeros(opt.misc.num_evaluations)

    train_err = np.zeros((opt.misc.num_evaluations, opt.t))
    valid_err = np.zeros((opt.misc.num_evaluations, opt.t))
    test_err = np.zeros((opt.misc.num_evaluations, opt.t))
    robust_err = np.zeros((opt.misc.num_evaluations, opt.t))
    correction_freq = np.zeros(opt.misc.num_evaluations)

    trajs_train = []
    for i in range(opt.iters):
        print "\nIteration: " + str(i)
        states, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t, False, False)
        trajs_train.append(states)
        lnr.add_data(states, int_actions)

        if (i + 1) % (opt.iters/opt.misc.num_evaluations) == 0:
            print "\tEvaluating..."
            lnr.train()
            fit_all(ocs, trajs_train)
            trajs_valid = []
            trajs_test = []
            trajs_robust = []

            sup_iters_rewards = np.zeros(opt.samples)
            lnr_iters_rewards = np.zeros(opt.samples)
            rob_iters_rewards = np.zeros(opt.samples)
            freqs = np.zeros(opt.samples)

            for j in range(opt.samples):
                print "\t\tSample: " + str(j) + " rolling out..."
                states_valid, int_actions_valid, _, r_valid = statistics.collect_traj(opt.env, opt.sup, opt.t, False, False)
                states_test, int_actions_test, _, r_test, _, lnr_score = statistics.collect_score_traj_multiple(opt.env, lnr, ocs, opt.t, False, False)
                states_robust, int_actions_robust, _, r_robust, freq, rob_score = statistics.collect_robust_traj_multiple(opt.env, lnr, ocs, opt.t, False, False)
                print "\t\tDone rolling out"

                trajs_valid.append(states_valid)
                trajs_test.append(states_test)
                trajs_robust.append(states_robust)

                sup_iters_rewards[j] = r_valid
                lnr_iters_rewards[j] = r_test
                rob_iters_rewards[j] = r_robust

                freqs[j] = freq

            index = i / (opt.iters / opt.misc.num_evaluations)
            train_err[index, :] = eval_ocs(ocs, trajs_train)
            valid_err[index, :] = eval_ocs(ocs, trajs_valid)
            test_err[index, :] = eval_ocs(ocs, trajs_test)
            robust_err[index, :] = eval_ocs(ocs, trajs_robust)

            sup_reward[index] = np.mean(sup_iters_rewards)
            lnr_reward[index] = np.mean(lnr_iters_rewards)
            rob_reward[index] = np.mean(rob_iters_rewards)

            correction_freq[index] = np.mean(freqs)




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




