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
samples = 100

obs = gen_model(M,K,D,samples)

#test = GMMHMM(n_components=5, n_mix=4, init_params='', params='mtcw', covariance_type='full', verbose=True, n_iter=100, min_covar=1e-7)
test = GMMHMM(n_components=5, n_mix=4, init_params='', params='mtcw', covariance_type='diag', verbose=False, n_iter=1, min_covar=1e-7)
test.startprob_ = np.ones((M)) / M
test.means_ = np.array((60 * np.random.random_sample((M, K, D)) - 30), dtype=np.double)
atmp = np.random.random_sample((M, M))
row_sums = atmp.sum(axis=1)
test.transmat_ = np.array(atmp / row_sums[:, np.newaxis], dtype=np.double)

wtmp = np.random.random_sample((M, K))
row_sums = wtmp.sum(axis=1)
test.weights_ = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

cv = np.cov(obs[0:40].T) + 1e-7 * np.eye(D)
test.covars_ = np.zeros((M, K, D))
test.covars_[:] = np.diag(cv)

#print(test.covars_)
#test.fit(obs[0:40], [5,5,5,5,5,5,5,5])
#print(test.means_)
#print(test.covars_)
#test.fit(obs[1:41], [5,5,5,5,5,5,5,5])
#print(test.means_)
#print(test.covars_)

# These are good values
saveds = test.startprob_
savedt = test.transmat_
savedm = test.means_
savedc = test.covars_
savedw = test.weights_
for i in range(window, samples):
  frame = obs[i-window:i]

  # while Degenerate, increase covars
  while True:
    test.fit(frame, [5,5,5,5,5,5,5,5])
    if np.all(np.isfinite(test.covars_)):
      break
    print("Degenerate. Increasing covars by 10%")
    test.startprob_ = saveds
    test.transmat_ = savedt
    test.means_ = savedm
    test.weights_ = savedw
    savedc = savedc * 1.1
    test.covars_ = savedc

  if np.any(test.covars_ == 0):
    test.covars_ += 1e-7

  print(test.covars_)
  print(i-window)
#  assert(np.all(test.covars_ != 0))
#  print(test.score(frame))
  # These are good values
  saveds = test.startprob_
  savedt = test.transmat_
  savedm = test.means_
  savedc = test.covars_
  savedw = test.weights_


#test2 = GMMHMM(n_components=5, n_mix=4, init_params='', params='mtcw', covariance_type='diag', verbose=True, n_iter=100, min_covar=1e-7)
#test2.startprob_ = test.startprob_
#test2.means_ = test.means_
#test2.transmat_ = test.transmat_
#test2.weights_ = test.weights_
#test2.covars_ = test.covars_
##cv = np.cov(obs[1:41].T) + 1e-7 * np.eye(D)
##test.covars_ = np.zeros((M, K, D, D))
##test.covars_[:] = cv
#test2.fit(obs[1:41], [5,5,5,5,5,5,5,5])
##print(test.means_)
##cv = np.cov(obs[2:42].T) + 1e-7 * np.eye(D)
##test.covars_ = np.zeros((M, K, D, D))
##test.covars_[:] = cv
##test.fit(obs[2:42], [5,5,5,5,5,5,5,5])
##print(test.means_)
