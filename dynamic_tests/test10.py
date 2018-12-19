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
    oc = svm.OneClassSVM(kernel='rbf', gamma = .05, nu = .05)
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)
    opt.samples = 100

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
        lnr.add_data(states, int_actions)
        trajs_train.append(states)


    lnr.train()
    print "\nCollecting validation samples..."
    trajs_valid = []
    trajs_test = []
    for j in range(opt.samples):
        states_valid, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t, False)
        states_test, int_actions, taken_actions, r = statistics.collect_traj(opt.env, lnr, opt.t, False, early_stop = False)
        trajs_valid.append(states_valid)
        trajs_test.append(states_test)
    print "Done collecting samples"

    X_train = []
    for traj in trajs_train:
        X_train += traj

    oc.fit(X_train)

    train_errs = np.zeros(opt.t)
    valid_errs = np.zeros(opt.t)
    test_errs = np.zeros(opt.t)
    adver_errs = np.zeros(opt.t)

    for t in range(opt.t):
        X_train = []
        for traj in trajs_train:
            X_train.append(traj[t])

        X_valid = []
        for traj in trajs_valid:
            X_valid.append(traj[t])

        X_test = []
        for traj in trajs_test:
            X_test.append(traj[t])

        X_train = np.array(X_train)
        cov = np.cov(X_train.T)
        mean = np.mean(X_train, axis=0)
        X_adver = np.random.multivariate_normal(mean, cov, opt.samples)

        train_err = eval_oc(oc, X_train)
        valid_err = eval_oc(oc, X_valid)
        test_err = eval_oc(oc, X_test)
        adver_err = eval_oc(oc, X_adver)
        print "Train Error: " + str(train_err)
        print "Valid Error: " + str(valid_err)
        print "Test Error: " + str(test_err)
        print "Adver Error: " + str(adver_err)
        print "Support vectors: " + str(oc.support_vectors_.shape)
        print "\n"

        train_errs[t] = train_err
        valid_errs[t] = valid_err
        test_errs[t] = test_err
        adver_errs[t] = adver_err


    plt.style.use('ggplot')
    #errs = [train_errs, valid_errs, test_errs]
    #labels = ['Training', 'Validation', 'Test']
    errs = [train_errs, valid_errs]
    labels = ['Training', 'Validation']


    width = .2
    index = np.arange(opt.t)


    for i, (err, label) in enumerate(zip(errs, labels)):
        plt.bar(index + i * width, err, width, label=label)
    plt.legend()
    plt.ylim(0, .75)
    plt.savefig('/Users/JonathanLee/Desktop/bar_single.png')
    utils.clear()






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




