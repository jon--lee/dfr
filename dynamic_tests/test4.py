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
from test1 import plot
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

    args = vars(ap.parse_args())
    opt = Options()

    opt.envname = args['env']
    opt.trials = args['trials']
    opt.iters = args['iters']
    opt.epochs = args['epochs']
    opt.lr = args['lr']
    opt.arch = args['arch']
    opt.t = args['t']

    opt.filename = '/Users/JonathanLee/experts/' + opt.envname + '.pkl'
    opt.env = gym.envs.make(opt.envname).env
    opt.sim = gym.envs.make(opt.envname).env
    opt.pi = load_policy.load_policy(opt.filename)
    opt.sess = tf.Session()
    opt.sup = NetSupervisor(opt.pi, opt.sess)

    run_trial(opt)


def run_trial(opt):
    oc = svm.OneClassSVM(kernel='rbf', nu=.01, gamma = .01)
    est = knet.Network(opt.arch, learning_rate = opt.lr, epochs = opt.epochs)
    lnr = learner.Learner(est)

    plot_dir = utils.generate_plot_dir('initial', 'experts', vars(opt))
    data_dir = utils.generate_data_dir('initial', 'experts', vars(opt))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    opt.plot_dir = plot_dir
    opt.data_dir = data_dir

    opt.num_valid_trajs = max(1, int(.25 * opt.iters))
    opt.samples = 10

    train_trajs = []
    valid_trajs = []

    for i in range(opt.iters):
        print "Iteration: " + str(i)
        states, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t)

        lnr.add_data(states, int_actions)

    oc.fit(lnr.X)
    preds = oc.predict(lnr.X)
    train_err = len(preds[preds == -1]) / float(len(preds))
    print "\nTraining error: " + str(train_err)




    lnr.train()

    sup_rewards = np.zeros((20))
    lnr_rewards = np.zeros((20))

    X_valid = []
    X_test = []
    for i in range(20):
        states_valid, int_actions_valid, _, r_valid = statistics.collect_traj(opt.env, opt.sup, opt.t, False)
        states_test, int_actions_test, _, r_test = statistics.collect_traj(opt.env, lnr, opt.t, False)

        sup_rewards[i] = r_valid
        lnr_rewards[i] = r_test
        
        X_valid += states_valid
        X_test += states_test

    valid_preds = oc.predict(X_valid) 
    valid_err = len(valid_preds[valid_preds == -1]) / float(len(valid_preds))
    print "Validation erorr: " + str(valid_err)

    test_preds = oc.predict(X_test)
    test_err = len(test_preds[test_preds == -1]) / float(len(test_preds))
    print "Test erorr: " + str(test_err)

    print "\n\n"

    print "Average sup reward: " + str(np.mean(sup_rewards)) + " +/- " + str(scipy.stats.sem(sup_rewards))
    print "Average lnr reward: " + str(np.mean(lnr_rewards)) + " +/- " + str(scipy.stats.sem(lnr_rewards))


    print "\n\n"


    def dec(u):
        x = opt.env.get_x()
        s, _, _, _ = opt.env.step(u)
        opt.env.set_x(x)
        return oc.decision_function([s])[0, 0]


    rewards = np.zeros((20))
    rec_counts = np.zeros((20))
    X_robust = []
    for i in range(20):

        s = opt.env.reset()
        states = [s]
        
        for t in range(opt.t):
            score = oc.decision_function([s])[0, 0]
            # print "Decision score: " + str(score)
            if score < .1:
                alpha = .1 
                a = alpha * utils.finite_diff1(np.zeros(opt.env.action_space.shape), dec)
                # print "Recovering: " + str(a)
                rec_counts[i] += 1.0
                s, r, done, _ = opt.env.step(a)
            else:
                a = lnr.intended_action(s)
                s, r, done, _ = opt.env.step(a)

            rewards[i] += r
            states.append(s)

            # if done == True:
            #     break

        X_robust += states

    robust_preds = oc.predict(X_robust)
    robust_err = len(robust_preds[robust_preds == -1]) / float(len(robust_preds))
    print "Robust erorr: " + str(robust_err)

    rec_freq = np.mean(rec_counts / float(opt.t))
    print "Recovery frequency: " + str(rec_freq)

    print "Robust rewards: " + str(np.mean(rewards)) + " +/- " + str(scipy.stats.sem(rewards))


def plot(datas, labels, opt, title):
    plt.style.use('ggplot')
    x = list(range(datas[0].shape[1]))

    for data, label in zip(datas, labels):
        mean = statistics.mean(data)
        ste = statistics.ste(data)

        plt.plot(x, mean, label = label)
        plt.fill_between(x, mean - ste, mean + ste, alpha=.3)
    
    plt.savefig(opt.plot_dir + title + "_plot.png") 
    utils.clear()


if __name__ == '__main__':
    main()




