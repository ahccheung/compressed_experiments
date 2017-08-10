from __future__ import division
import sys
import csv
import numpy as np

def evaluate(X, V, weights):
    return np.apply_along_axis(lambda x: from_list(x, V, weights), 1, X)

def from_list(x,V,weights):
    # V = L x order X J
    L = V.shape[0]
    f = lambda i: weights[i]*inner_from_vector(x, V[i])
    return np.sum(map(f, np.arange(0,L)))

def inner_from_vector(x, Vi):
    # Vi = order x J
    ip = np.apply_along_axis(lambda v: np.inner(v,x), 1, Vi)
    prod = np.product(ip)
    return prod

def inner(x_tensor, tensor):
    return np.sum(np.multiply(x_tensor, tensor))

def tensor_from_list(x_list):
    return np.multiply.reduce(np.ix_(*x_list))

def eval_tensor(x, tensor):
    order = len(tensor.shape)
    x_list =[x]*order
    x_tensor = tensor_from_list(x_list)
    return inner(x_tensor, tensor)

def shallow_ys(X, activities, T):
    #X = N x J, Y = N x G, activities = 1 x G
    N = X.shape[0]
    hidden = np.apply_along_axis(lambda x: eval_tensor(x, T), 1, X)
    Y = np.array(map(lambda y: np.multiply( y, activities), hidden))
    return Y

def random_rank1(dim, order,k):
    V = []
    for n in range(order):
        v = abs(np.random.randn(dim))
        if k > -1:
            xs = np.argsort(abs(v)) #vector from 0 to n-1
            v[xs[:-k]] = 0 # set items with smallest abs value to 0
        v = v/np.linalg.norm(v)
        V.append(v)
    T = np.multiply.reduce(np.ix_(*V))
    # T  tensor
    # V = list of order vectors of length dim each
    return V,T

def generate_tensor(dim, order, k, L):
    """"L = rank, k = sparsity."""

    weights = abs(np.random.randn(L))
    #weights[weights < 0.1] = 0
    V0 = np.empty((L,order,dim)) # 3 x 4 x 50 tensor
    T0 = np.zeros(shape = np.repeat(dim, order)) # tensor, dim^order

    for i in range(L):
        V,T = random_rank1(dim,order,k)
        T0 += weights[i]*T
        V0[i] = V
    return V0,weights,T0

def generate_X(N, J):
    """ X = N X J random matrix."""

    X = abs(np.random.randn(N,J))
    X[X<0.2] = 0
    return X

def shallow_data(N, J, G, order, k, L):
    X = generate_X(N,J)
    V,weights,T = generate_tensor(J, order, k, L)
    activities = abs(np.random.randn(G))
    activities[activities < 0.1] = 0
    #biases = abs(np.random.randn(G))
    #biases[biases<0.5] = 0
    Y = shallow_ys(X, activities, T)

    return X, Y, activities, V, weights, T

def main(N, J, G, order, k, L, flabel):

    X, Y, a, V, w, T = shallow_data(N,J,G,order,k,L)
    test_size = N//5
    train_size = N - test_size


    with open(flabel + 'act', 'ab') as csvfile:
        act = csv.writer(csvfile, delimiter=' ', quotechar='|')
        act.writerow(a)

    with open(flabel + 'T', 'ab') as csvfile:
        v = csv.writer(csvfile, delimiter=' ', quotechar='|')
        for l in range(L):
            v.writerow([w[l]])
            for o in range(order):
                v.writerow(V[l,o])

    train_x = open(flabel + '_trainx', 'ab')
    w_trainx = csv.writer(train_x)
    train_y = open(flabel + '_trainy', 'ab')
    w_trainy = csv.writer(train_y)

    for i in range(train_size):
        w_trainx.writerow(X[i])
        w_trainy.writerow(Y[i])
    train_x.close()
    train_y.close()

    test_x = open(flabel + '_testx', 'ab')
    w_testx = csv.writer(test_x)
    test_y = open(flabel + '_testy', 'ab')
    w_testy = csv.writer(test_y)

    for j in range(train_size, N):
        w_testx.writerow(X[j])
        w_testy.writerow(Y[j])
    test_x.close()
    test_y.close()

if __name__=="__main__":
    [N, J, G, order, k, L]= map(int, sys.argv[1:-1])
    flabel = sys.argv[-1]
    main(N, J, G, order, k, L, flabel)
