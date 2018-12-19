import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

opts = pickle.load(open('opt_dvrk.pkl', 'r'))

scores_data = np.array(opts['rec_scores'])
cutoffs_data = np.array(opts['rec_cutoffs'])

values = (scores_data / cutoffs_data)[1:30]
print values.shape
values = np.clip(values, 0, 1)
means = np.mean(values, axis=0)
stds = scipy.stats.sem(values, axis=0)

print values.shape

# plt.style.use('ggplot')

x = np.arange(means.shape[0])
plt.plot(x, means, color='#328ABD', label='DFR', linewidth=4)
for value in values:
    plt.plot(x, value, color='blue', alpha=.5)

plt.fill_between(x, means - stds, means + stds, color='#328ABD', alpha=.5)
# plt.errorbar(x, means, stds, ecolor='black', capthick=1, elinewidth=1, color='#328ABD', capsize=5, errorevery=2)
# plt.plot(x, np.ones(means.shape), color='black', linestyle='dashed')
plt.ylim(.4, 1.1)
plt.title("Line Tracking", fontsize=25)
plt.ylabel('Normalized Decision Function', fontsize=22)
plt.xlabel('Iterations', fontsize=22)
plt.legend(loc='lower right', fontsize=20)
plt.show()
# plt.savefig('tmp_plots/opt_dvrk.pdf')



