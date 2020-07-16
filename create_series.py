import random
import numpy as np
from hmmlearn.hmm import GMMHMM
import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(100)

M = 5
K = 4
D = 2
model = GMMHMM(n_components=5, n_mix=4, init_params='', params='stmcw', covariance_type='diag')
model.startprob_ = np.array([1,0,0,0,0])
atmp = np.random.random_sample((M, M))
row_sums = atmp.sum(axis=1)
model.transmat_ = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)

wtmp = np.random.random_sample((M, K))
row_sums = wtmp.sum(axis=1)
model.weights_ = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

model.means_ = np.array((0.6 * np.random.random_sample((M, K, D)) - 0.3), dtype=np.double)
model.covars_ = np.ones( (M,K,D) )

obs, vit = model.sample(200)

print(model.score(obs))
test = GMMHMM(n_components=5, n_mix=4, init_params='smtcw', params='smtcw', covariance_type='diag', verbose=True)
# test.means_ = model.means_
test.fit(obs)
print(test.score(obs))
