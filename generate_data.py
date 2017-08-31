from __future__ import division
import sys
import csv
import numpy as np
from scipy import special

def data_to_file(X, Y, flabel):
  N = X.shape[0]
  test_size = N//5
  train_size = N - test_size
  idx = np.arange(N)
  np.random.shuffle(idx)

  train_x = flabel + '_trainx'
  test_x = flabel + '_testx'
  train_y = flabel + '_trainy'
  test_y = flabel + '_testy'

  np.savetxt(train_x, X[idx[:train_size]], delimiter=',')
  np.savetxt(train_y, Y[idx[:train_size]], delimiter=',')

  np.savetxt(test_x, X[idx[train_size:]], delimiter=',')
  np.savetxt(test_y, Y[idx[train_size:]], delimiter=',')

def tensor_from_list(x_list):
  return np.multiply.reduce(np.ix_(*x_list))

def shallow_ys(X, offsets, module_effects, biases, V, weights):
  #X = N x J, Y = N x G, offsets = 1xL

  N = X.shape[0]
  f = lambda x: eval_tensor(x, offsets, V, weights)
  hidden = np.apply_along_axis(f, 1, X)

  Y_lambda= np.array([np.multiply(h, module_effects)+biases for h in  hidden])
  Y_lambda = Y_lambda.astype('int')
  Y = poisson(Y_lambda)
  return Y

def poisson(lda_vec):
  return np.apply_along_axis(lambda lda: np.random.poisson(lda), 0, lda_vec)

def random_rank1(dim, order,k):
  # T  tensor
  # V = list of order vectors of length dim each
  V = []
  for n in range(order):
    v = abs(np.random.randn(dim))
    if k > -1:
      l = np.random.poisson(k/2)
      if l>k: l = k
      if l==0: l=1
      xs = np.argsort(abs(v)) #vector from 0 to n-1
      v[xs[:-k]] = 0 # set items with smallest abs value to 0
    v = v/np.linalg.norm(v)
    V.append(v)
  T = np.multiply.reduce(np.ix_(*V))

  return V,T

def generate_tensor(dim, order, k, L):
  """"L = rank, k = sparsity."""

  sum_weights = abs(np.random.randn(L))
  V0 = np.empty((L,order,dim)) # 3 x 4 x 50 tensor
  T0 = np.zeros(shape = np.repeat(dim, order)) # tensor, dim^order

  for i in range(L):
    V,T = random_rank1(dim,order,k)
    T0 += sum_weights[i]*T
    V0[i] = V
  return V0,sum_weights,T0

def generate_poisson_vector(J, exp_lambda):
  return np.random.randint(exp_lambda, size = (J))

def generate_X(N, J, s, exp_lambda):
  """ X = N X J random matrix.
  s denotes the number of samples with same poisson parameters."""
  samples_generated = 0
  X = np.zeros((0, J))

  while samples_generated < N:
    lda_vec = generate_poisson_vector(J, exp_lambda)
    n = s
    if N - (samples_generated + s) < s:
      n = N - samples_generated
    samples = np.array([poisson(lda_vec) for i in range(n)])
    X = np.vstack((X, samples))
    samples_generated = samples_generated + n

  return X

def generate_offsets(X,V,L):
  pooled = np.array([pool(x, V) for x in X])
  offset = np.mean(pooled, axis = 0)
  return offset

def generate_module_effects(G, exp_lambda):

  mod_effects = abs(np.random.normal(0, exp_lambda, G))
  mod_effects[mod_effects < 0.1] = 0
  biases = np.random.poisson(exp_lambda//2, size = (G))
  biases[biases<=exp_lambda//3] = 0

  return mod_effects, biases

def relu(x, offsets):
  f = lambda x: max(0, x)
  translated = x - offsets
  return np.array([f(translate) for translate in translated])

def pool(x, V):
  conv = np.array([v.dot(x) for v in V])
  products = np.apply_along_axis(np.multiply.reduce, 1, conv)
  return products

def eval_tensor(x, offsets, V, weights):
  #conv = L x order, products = L x 1
  products = pool(x, V)
  activations = relu(products, offsets)
  return np.inner(weights, activations)

def model_to_file(mod_effects, biases, offsets, weights, V, flabel):
  J = V.shape[2]
  L = V.shape[0]

  np.savetxt(flabel + 'offsets', offsets, delimiter=',')

  with open(flabel + 'module_effects', 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',', quotechar='|')
    w.writerow(mod_effects)
    w.writerow(biases)

  with open(flabel + 'T', 'wb') as csvfile:
    v = csv.writer(csvfile, delimiter=',', quotechar='|')
    v.writerow(weights)
    for j in range(J):
      for l in range(L):
        v.writerow(V[l,:,j])

def main(N, J, G, order, k, L, s,flabel, exp_lambda):

  X = generate_X(N,J,s, exp_lambda)
  V, weights, T = generate_tensor(J, order, k, L)
  mod_effects, biases = generate_module_effects(G, exp_lambda)
  offsets = generate_offsets(X,V, L)
  Y = shallow_ys(X, offsets, mod_effects, biases, V, weights)
  model_to_file(mod_effects, biases, offsets,weights, V, flabel)
  data_to_file(X, Y, flabel)

if __name__=="__main__":
  [N, J, G, order, k, L, s]= map(int, sys.argv[1:-1])
  flabel = sys.argv[-1]
  exp_lambda = 15
  main(N, J, G, order, k, L, s, flabel, exp_lambda)
