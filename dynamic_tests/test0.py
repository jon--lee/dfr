"""
Linear quadratic regulator on Mujoco domains
"""

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
from icra_tools import statistics, utils
from icra_tools.expert import load_policy
from icra_tools import tracking_controller
from icra_tools.supervisor import NetSupervisor

def load_env(envname):
    filename = '/Users/JonathanLee/experts/' + envname + '.pkl'
    env = gym.envs.make(envname).env
    pi = load_policy.load_policy(filename)
    sess = tf.Session()
    sup = NetSupervisor(pi, sess)
    return env, sup


def evaluate(env, contr, T, x_array, u_array):
    env.reset()
    env.reset_x()
    env.render()
    x = env.get_x()
    reward = 0.0
    for t in range(T - 1):
        env.render()
        u = contr.contr(x, t, x_ref = x_array[t], u_ref = u_array[t])

        _, _, r, _ = env.step(u)
        x = env.get_x()
        reward += r

    print "Reward: " + str(reward)


if __name__ == '__main__':
    envname = 'Walker2d-v1'
    env, sup = load_env(envname)
    simulator = gym.envs.make(envname).env
    env.reset()
    simulator.reset()
    env.reset_x()
    simulator.reset_x()
    T = 1000

    contr = tracking_controller.Controller()

    x_array, u_array, taken_actions, r = statistics.collect_traj_alt(env, sup, T, visualize=True)
    print "Supervisor reward: " + str(r)


    Cs, Fs, cs, fs = contr.linearize(env, simulator, x_array, u_array)
    # approximations = pickle.load(open('approx.pkl', 'w'))
    # Cs, Fs, cs, fs = approximations
    approximations = [Cs, Fs, cs, fs]
    pickle.dump(approximations, open("approx.pkl", 'w'))



    Vs, vs, Ks, ks = contr.backups(env, T, Cs, Fs, cs, fs)

    evaluate(env, contr, T, x_array, u_array)
    IPython.embed()



