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

test = GMMHMM(n_components=5, n_mix=4, init_params='tcw', params='mtcw', covariance_type='diag', verbose=True)
test.startprob_ = np.ones((M)) / M
test.means_ = np.array((0.6 * np.random.random_sample((M, K, D)) - 0.3), dtype=np.double)

for i in range(window, samples):
  frame = obs[i-window:i]
  test.fit(frame)
  print(test.means_)
  test2 = GMMHMM(n_components=5, n_mix=4, init_params='tcw', params='mtcw', covariance_type='diag', verbose=True)
  test2.means_ = test.means_
  test2.startprob_ = test.startprob_
  test = test2
#  print(test.covars_)
#  print(test.startprob_)
#print(test.score(obs))
