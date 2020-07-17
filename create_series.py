import random
import numpy as np
from hmmlearn.hmm import GMMHMM
import sys
from genmodel import gen_model
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(100)

M = 5
K = 4
D = 2
window = 40
samples = 400

obs = gen_model(M,K,D,samples)


test = GMMHMM(n_components=5, n_mix=4, init_params='', params='mtcw', covariance_type='diag', verbose=False, n_iter=100, min_covar=1e-7)
test.startprob_ = np.ones((M)) / M
test.means_ = np.array((0.6 * np.random.random_sample((M, K, D)) - 0.3), dtype=np.double)
atmp = np.random.random_sample((M, M))
row_sums = atmp.sum(axis=1)
test.transmat_ = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)

wtmp = np.random.random_sample((M, K))
row_sums = wtmp.sum(axis=1)
test.weights_ = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

cv = np.cov(obs[0:40].T) + 1e-7 * np.eye(D)
test.covars_ = np.zeros((M, K, D))
test.covars_[:] = np.diag(cv)
covars_orig = test.covars_


for i in range(window, samples):
  frame = obs[i-window:i]
  test.fit(frame)
#  print(test.means_)
  test.covars_ = test.covars_ * 10 + 1e-7
  print(test.covars_)
