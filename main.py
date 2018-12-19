import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import matplotlib.pyplot as plt
from icra_tools.supervisor import NetSupervisor
import tensorflow as tf
import numpy as np
from icra_tools.expert import load_policy
from icra_tools import statistics
import IPython
from sklearn import svm
from icra_tools import utils

envname = 'Walker2d-v1'
filename = '/Users/JonathanLee/experts/' + envname + '.pkl'
env = gym.envs.make(envname)
pi = load_policy.load_policy(filename)
sess = tf.Session()
sup = NetSupervisor(pi, sess)

T = 1000
ITERATIONS = 10
trajs_train = []
trajs_test = []

for i in range(7):
    print "{ Iteration: " + str(i) + " }"
    s = env.reset()
    reward = 0.0
    traj = []
    for t in range(T):
        # env.render()
        a = sup.sample_action(s)
        next_s, r, done, _ = env.step(a)

        if done:
            break

        traj.append((s, a))

        reward += r
        s = next_s

    print "Reward: " + str(reward)
    trajs_train.append(traj)


for i in range(3):
    print "{ Iteration: " + str(i) + " }"
    s = env.reset()
    reward = 0.0
    traj = []
    for t in range(T):
        # env.render()
        a = sup.sample_action(s)
        next_s, r, done, _ = env.step(a)

        if done:
            break

        traj.append((s, a))

        reward += r
        s = next_s

    print "Reward: " + str(reward)
    trajs_test.append(traj)



clf = svm.OneClassSVM(nu=.05, kernel='rbf', gamma=.02)
X_train, y_train = utils.trajs2data(trajs_train)
X_test, y_test = utils.trajs2data(trajs_test)

clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

error_train = y_pred_train[y_pred_train == -1].size / float(len(y_pred_train))
error_test = y_pred_test[y_pred_test == -1].size / float(len(y_pred_test))

print "Train error: " + str(error_train)
print "Valid error: " + str(error_test)


mean = np.mean(X_test, axis=0)
var = np.cov(X_test.T)

samples = np.random.multivariate_normal(mean, var, 500)
preds = clf.predict(samples)
error = preds[preds == -1].size / float(len(preds))

print "Test errr: " + str(error)





