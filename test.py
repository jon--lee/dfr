import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn.svm as svm
import IPython
from scipy.stats import multivariate_normal

X1 = np.random.multivariate_normal([2, 2], .1 * np.identity(2), 100)
X2 = np.random.multivariate_normal([0, 2], .1 * np.identity(2), 200)
X3 = np.random.multivariate_normal([2, 3], .1 * np.identity(2), 200)

X = np.vstack((X1, X2, X3))

# lower gamma is more elliptical



gamma = .000001
nu = .75

oc = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
oc.fit(X)

plt.style.use('ggplot')



xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
zeros = np.zeros((xx.shape[0] * xx.shape[1], 2))
x = xx.reshape(xx.shape[0] * xx.shape[1])
y = yy.reshape(yy.shape[0] * yy.shape[1])
zeros[:, 0] = x
zeros[:, 1] = y
values = oc.decision_function(zeros).reshape((xx.shape[0], xx.shape[1]))

decisions = oc.predict(zeros).reshape((xx.shape[0], xx.shape[1]))

plt.contourf(xx, yy, values, 20)
plt.contour(xx, yy, decisions, 20, colors='blue')


preds = oc.predict(X)
error = len(preds[preds == -1]) / float(len(preds))
print "Error: " + str(error)


plt.scatter(X[:, 0], X[:, 1])
plt.ylim(0, 4)
plt.xlim(-4, 4)
plt.axhline(y=0, color="#888888", linestyle='--')
plt.axvline(x=0, color="#888888", linestyle='--')
plt.show()

