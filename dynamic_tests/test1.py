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
    oc = svm.OneClassSVM(kernel='rbf', nu=.01)
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

    sup_rewards = np.zeros((1, opt.iters))
    lnr_rewards = np.zeros((opt.samples, opt.iters))

    sup_perf = np.zeros((1, opt.iters))
    lnr_perf = np.zeros((opt.samples, opt.iters))

    for i in range(opt.iters):
        print "Iteration: " + str(i)
        states, int_actions, taken_actions, r = statistics.collect_traj(opt.env, opt.sup, opt.t)
        sup_rewards[0, i] = r
        sup_perf[0, i] = opt.env.metric()

        lnr.add_data(states, int_actions)
        lnr.train()

        print "\t" + str(lnr.acc())
        for j in range(opt.samples):
            _, _, _, r = statistics.collect_traj(opt.env, lnr, opt.t)
            lnr_rewards[j, i] = r
            lnr_perf[j, i] = opt.env.metric()


    print "Average success: " + str(sup_rewards)
    print "Learner success: \n" + str(lnr_rewards)

    pd.DataFrame(sup_rewards).to_csv(opt.data_dir + 'sup_rewards.csv')
    pd.DataFrame(lnr_rewards).to_csv(opt.data_dir + 'lnr_rewards.csv')
    pd.DataFrame(sup_perf).to_csv(opt.data_dir + 'sup_perf.csv')
    pd.DataFrame(lnr_perf).to_csv(opt.data_dir + 'lnr_perf.csv')

    plot([sup_rewards, lnr_rewards], ['sup', 'lnr'], opt, 'Reward')
    plot([sup_perf, lnr_perf], ['sup', 'lnr'], opt, 'Performance')


def plot(datas, labels, opt, title):
    plt.style.use('ggplot')
    x = list(range(datas[0].shape[1]))
    for data, label in zip(datas, labels):
        mean = statistics.mean(data)
        ste = statistics.ste(data)
        plt.plot(x, mean, label = label)
        plt.fill_between(x, mean - ste, mean + ste, alpha=.3)
    plt.legend()
    plt.show()
    # plt.savefig(opt.plot_dir + title + "_plot.png") 
    # utils.clear()


if __name__ == '__main__':
    main()




