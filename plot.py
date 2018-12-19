import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import IPython
I = 300
rec_failed = np.array([0, 0, 1])
lnr_failed = np.array([0, 0, 0])
rec_false_positives = np.array([224, 98, 184])
lnr_false_positives = np.array([247, 164, 237])
rec_true_positives = np.array([14, 5, 5])
lnr_true_positives =np.array( [22, 5, 7])
rec_false_negatives =np.array( [37, 15, 17])
lnr_false_negatives = np.array([26, 18, 6])
rec_failures = np.array([51, 20, 25])
lnr_failures = np.array([26, 23, 15])


rec_data = [rec_failed, rec_failures, rec_false_positives, rec_true_positives, rec_false_negatives]
lnr_data = [lnr_failed, lnr_failures, lnr_false_positives, lnr_true_positives, lnr_false_negatives]
rec_data = np.array(rec_data) / float(I)
lnr_data = np.array(lnr_data) / float(I)

rec_means = [np.mean(bar) for bar in rec_data]
rec_sems = [stats.sem(bar) for bar in rec_data]

lnr_means, lnr_sems = [np.mean(bar) for bar in lnr_data], [stats.sem(bar) for bar in lnr_data]

rec_means.append(1 - sum(rec_means[2:]))
lnr_means.append(1 - sum(lnr_means[2:]))

plt.style.use('ggplot')
labels = ['caused fail', 'failures', 'false pos', 'true pos', 'false neg', 'true neg']

IPython.embed()

width = .4
index = np.arange(len(rec_means))
i = 0
# for i, (err, label) in enumerate(zip(bar_errs, labels)):
#     plt.bar(index + i * width, err, width, label=label)
plt.bar(index + 0 * width, rec_means, width, label='recovery')
plt.bar(index + 1 * width, lnr_means, width, label='no recovery')
plt.legend()
plt.xticks(index, labels)
plt.ylim(0, 1)
plt.show()
