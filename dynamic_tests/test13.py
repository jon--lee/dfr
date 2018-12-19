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
from test12 import make_bar_graphs
from test12 import fit_all

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

    train_errs = np.zeros((opt.trials, opt.t))
    valid_errs = np.zeros((opt.trials, opt.t))
    test_errs = np.zeros((opt.trials, opt.t))

    for t in range(opt.trials):
        train_errs[t, :], valid_errs[t, :], test_errs[t, :] = run_trial(opt, t)

    train_err = np.mean(train_errs, axis=0)
    valid_err = np.mean(valid_errs, axis=0)
    test_err = np.mean(test_errs, axis=0)

    errs = [train_err, valid_err, test_err]
    labels = ['train', 'valid', 'test']

    width = .2
    index = np.arange(opt.t)

    for i, (err, label) in enumerate(zip(errs, labels)):
        plt.bar(index + i * width, err, width, label=label)
    plt.legend()
    plt.ylim(0, .75)
    plt.savefig('/Users/JonathanLee/Desktop/bar_new_avg.png')
    utils.clear()

def eval_oc(oc, X):
    preds = oc.predict(X)
    err = len(preds[preds == -1]) / float(len(preds))
    return err

def run_trial(opt, trial_count):
    ocs = [ svm.OneClassSVM(kernel='rbf', gamma = .05, nu = .05) for t in range(opt.t) ]
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)
    opt.samples = 100

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

    fit_all(ocs, trajs_train)

    errs = make_bar_graphs(ocs, trajs_train, trajs_valid, trajs_test, opt, filename ='/Users/JonathanLee/Desktop/bar_new/bar_new' + str(trial_count) + '.png')
    return errs



if __name__ == '__main__':
    main()




