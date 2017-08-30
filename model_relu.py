import os
import getpass
import sys
import time
import csv
import operator
import numpy as np
import tensorflow as tf
import model_metrics as met

class Config(object):
  """Holds model hyperparams and data information.
  """
  batch_size = 100
  rank = 1
  order = 3
  max_epochs = 500
  lr = 0.01
  lda1 = 20
  lda2 = 0.01

  def __init__(self, f_lab, snr=None):
    self.random_init = (snr == None)
    if snr != None:
      self.add_noise = (np.isfinite(snr))
    else: self.add_noise = False
    self.f_lab = f_lab
    self.snr = snr

class ShallowModel(object):
  """Implements model for shallow decomposition
  """

  def load_data(self, f_lab):
    """Loads data from f_lab_trainx, f_lab_trainy.
    """
    self.x_train = np.loadtxt(f_lab + '_trainx', delimiter=',', dtype='float32')
    self.y_train = np.loadtxt(f_lab + '_trainy', delimiter=',',dtype ='float32')
    self.n_samples = self.x_train.shape[0]
    self.input_size = self.x_train.shape[1]
    self.output_size = self.y_train.shape[1]

    self.x_test = np.loadtxt(f_lab + '_testx', delimiter=',', dtype = 'float32')
    self.y_test = np.loadtxt(f_lab + '_testy', delimiter=',', dtype = 'float32')

  def add_placeholders(self):
    """Adds placeholder variables to tensorflow computational graph.
    """
    self.x = tf.placeholder(tf.float32,[None, self.input_size])
    self.y = tf.placeholder(tf.float32,[None, self.output_size])

  def create_feed_dict(self, in_batch, out_batch):
    """Creates the feed_dict for training the given step.
    Args:
      in_batch: A batch of input data.
      out_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    feed_dict = {self.x: in_batch, self.y: out_batch}
    return feed_dict

  def true_model(self):
    """Loads information about simulated data to get weights for true model.
    """

    with open(self.config.f_lab + 'T', 'rb') as csvfile:
      r = csv.reader(csvfile, delimiter=' ')
      w = float(r.next()[0])
      self.conv_true = w * np.array([row for row in r]).astype('float32').T

    with open(self.config.f_lab + 'module_effects', 'rb') as csvfile:
      r = csv.reader(csvfile,delimiter = ' ')
      self.mod_true = np.array(r.next()).astype('float32')
      self.mod_true = self.mod_true.reshape((1, self.output_size))
      self.bias_true = np.array(r.next()).astype('float32')
      self.bias_true = self.bias_true.reshape((1, self.output_size))

    #self.pool_true = 10*np.ones(self.config.order)
    self.pool_true = np.zeros(self.config.order)
    with open(self.config.f_lab + 'activations', 'rb') as csvfile:
      r = csv.reader(csvfile, delimiter=' ')
      self.act_true = np.array(r.next()).astype('float32')
      self.k_true = np.array(r.next()).astype('float32')

  def get_variances(self):

    f = lambda x: np.mean(x)**2
    weights = [self.conv_true, self.mod_true, self.bias_true, self.act_true,self.pool_true, self.k_true]
    mu_2 = [f(x) for x in weights]
    g = lambda x: np.sqrt(x/self.config.snr)

    return [g(mu) for mu in mu_2]

  def initial_weights(self):
    """Initializes weights for models by adding noise to true weights.
    """

    if self.config.random_init:
      self.conv_init = np.random.normal(0, 1, (self.input_size,self.config.order)).astype('float32')
      self.mod_init = np.random.poisson(1, (1, self.output_size)).astype('float32')
      self.bias_init = np.random.poisson(1, (1, self.output_size)).astype('float32')
      self.act_init = np.random.poisson(1,(1, self.config.rank)).astype('float32')
      self.pool_init = np.zeros((1, self.config.order)).astype('float32')
      self.k_init = np.random.exponential(5, self.config.rank).astype('float32')

    elif not self.config.add_noise:
      self.conv_init = self.conv_true
      self.mod_init = self.mod_true
      self.bias_init = self.bias_true
      self.act_init = self.act_true
      self.pool_init = self.pool_true
      self.k_init = self.k_true

    else:
      var = self.get_variances()
      self.conv_init = (self.conv_true + np.random.normal(0, var[0],(self.input_size,self.config.order))).astype('float32')
      self.mod_init = (self.mod_true + abs(np.random.normal(0, var[1], (1, self.output_size)))).astype('float32')
      self.bias_init = (self.bias_true + abs(np.random.normal(0, var[2], (1, self.output_size)))).astype('float32')
      self.act_init = (self.act_true + np.random.normal(0, var[3], (1, self.config.rank))).astype('float32')
      self.pool_init = (self.pool_true + np.random.normal(0, var[4], (1, self.config.order))).astype('float32')
      self.k_init = abs(self.k_true + np.random.normal(0, var[5], (self.config.rank))).astype('float32')

  def add_model(self, input_data):
    """Implements core of model that transforms input_data into predictions.
    The core transformation for this model which transforms a batch of input
    data into a batch of predictions.

    Args:
      input_data: A tensor of shape (batch_size, input_size).
    Returns:
      pred: A tensor of shape (batch_size, output_size)
      W, pool_weights, activations, mod_effects, biases: weights of model
    """
    # Hidden
    with tf.name_scope('hidden'):
      W = tf.abs(tf.Variable(
        self.conv_init, name = 'Conv_weights'))
      conv = tf.matmul(self.x, W)

      pool_weights = tf.Variable(self.pool_init, dtype='float32',name = 'Pool_weights')
      pooled = self.pool(conv, pool_weights)

      activations = tf.Variable(self.act_init, name = 'Activations')
      self.k = tf.Variable(self.k_init, name='Log_steepness')
      sig = self.logistic(self.k, pooled - activations) * pooled

      mod_effects = tf.abs(tf.Variable(
               self.mod_init, name = 'Module_activity'))
      biases = tf.abs(tf.Variable(
             self.bias_init, name = 'Biases'))
      pred  = tf.matmul(sig, mod_effects) + biases
      self.sig =sig
    return pred, W, pool_weights, activations, mod_effects, biases

  def logistic(self, k, x):
    l = tf.reciprocal(1 + tf.exp(tf.multiply(-k, x)))
    return l

  def logistic_diff(self):
    a_fit = met.prediction_fit(tf.multiply(self.k, self.act_true) ,tf.multiply(self.k, self.activations))
    k_fit = met.prediction_fit(self.k_true,self.k)
    return (a_fit+ k_fit )/2

  def polynomial(self, sess, conv, pool_vec):
    weights = np.array(sess.run(tf.multiply(conv, pool_vec)).T)
    idx, vals = self.indices(weights, [],[])
    idx = self.sort_indices(idx)
    poly = self.to_tensor(idx, vals)
    poly = tf.convert_to_tensor(poly, dtype='float32')
    return poly

  def sort_indices(self,inds):
    return [sorted(ind) for ind in inds]

  def indices(self,weights, inds, vals):
    if weights.shape[0]==0:
      return inds, vals
    else:
      w = weights[0]
      winds = np.nonzero(w)[0]
      if inds ==[]:
        inds = [[i] for i in winds]
        vals = [w[i] for i in winds]
      else:
        inds = [[ind + [i] for i in winds] for ind in inds]
        inds = reduce(operator.add, inds)
        vals = [[val *w[i] for i in winds] for val in vals]
        vals = reduce(operator.add, vals)
      return self.indices(weights[1:], inds, vals)

  def to_tensor(self,inds, vals):
    dims = tuple(np.repeat(self.input_size, self.config.order))
    T = np.zeros(dims)
    for i in range(len(inds)):
      ind = tuple(inds[i])
      T.itemset(ind, vals[i])
    return T

  def model_fit(self, sess):
    pool_vec = tf.sigmoid(self.pool_weights)
    ones = tf.constant(1., shape=[self.config.order])
    true_poly = self.polynomial(sess, self.conv_true, ones)
    model_poly = self.polynomial(sess, self.W, pool_vec)

    poly_fit = met.prediction_fit(true_poly, model_poly)

    fit1 = self.logistic_diff()

    fit2 = (met.prediction_fit(self.mod_true, self.mod_effects) \
           + met.prediction_fit(self.bias_true,self.biases)) /2

    return sess.run(poly_fit), sess.run(fit1), sess.run(fit2)

  def pool(self, conv, pool_weights):
    """Continuous version of product pooling.
    Each input x is transformed to 1 - sigmoid(weight) + sigmoid(weight) * x.
    Eg: If tf.sigmoid(weight) = 0, get x -> 1
        If tf.sigmoid(weight) = 0.5, get x -> 0.5 + 0.5x
        If tf.sigmoid(weight) = 1, get x -> x
    Then transformed inputs are then product pooled."""

    pool_vec = tf.sigmoid(pool_weights)
    weighted = 1 - pool_vec + tf.multiply(conv, pool_vec)
    return tf.expand_dims(tf.cumprod(weighted, axis = 1)[:,-1],1)

  def add_loss_op(self, pred):
    """Adds ops for loss to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, output_size)
    Returns:
      loss: A 0-d tensor (scalar) output
    """
    reg = self.config.lda1 *self.config.batch_size* tf.norm(self.W, ord = 1) \
          + self.config.lda2 * tf.norm(tf.sigmoid(self.pool_weights), ord = 1)

    return tf.reduce_mean(tf.norm(self.y - pred, axis=1)**2)

  def run_epoch(self, sess, input_data, output_data):
    """Runs an epoch of training. Trains the model for one-epoch.

    Args:
      sess: tf.Session() object
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    """
    gva = tf.gradients(self.loss, self.activations)
    gvm = tf.gradients(self.loss, self.mod_effects)
    gvp = tf.gradients(self.loss, self.pool_weights)
    gvw = tf.gradients(self.loss, self.W)

    batches = self.batches(input_data, output_data,self.config.batch_size)
    for (x, y) in batches:
      feed = self.create_feed_dict(x, y)
      _ = sess.run([self.train_op], feed_dict=feed)

  def batches(self, X,  Y, batch_size):
    """"Separates data X,Y into batches of length batch_size.
    """
    batches = []
    count = 0

    while len(X[count:]) > 0:
      increment = min(batch_size, len(X[count:]))
      if np.any(Y):
        Ybatch = Y[count:count + increment]
      else:
        Ybatch = np.array([])
      batch = (X[count:count + increment], Ybatch)
      count += increment
      batches.append(batch)

    return batches

  def prune_weights(self, sess, W):
    W = sess.run(W)
    pruned = np.array([self.prune_col(col) for col in W.T]).T
    return tf.convert_to_tensor(pruned, dtype='float32')

  def prune_col( self,col):
    col_max = max(abs(col))
    prune = [30*abs(cell)< col_max and cell!= 0 for cell in col]
    if not np.any(prune):
      return col
    else:
      tmp = np.copy(col)
      tmp[prune] = 0
      return self.prune_col(tmp)

  def fit(self, sess):
    """Fit model on provided data.
    """
    data = []
    train_fd = self.create_feed_dict(self.x_train, self.y_train)
    test_fd =self.create_feed_dict(self.x_test, self.y_test)

    for i in range(self.config.max_epochs):
      if (i % 100 ==0):
        print 'Step %d:' %(i)
        metrics  = met.model_metrics(sess)
        data.append([i] + metrics)

      self.run_epoch(sess, self.x_train, self.y_train)
    print sess.run(self.W)
    pruned= self.prune_weights(sess, self.W)
    print sess.run(pruned)
    print self.conv_true
    metrics = met.model_metrics(sess)
    data.append([self.config.max_epochs] + metrics)
    return data

  def add_training_op(self, loss):
    """Sets up the training Ops.
    Args:
      loss: Loss tensor, from add_loss_op.
    Returns:
      train_op: The Op for training.
    """
    optimizer = tf.train.AdamOptimizer(self.config.lr)
    train_op = optimizer.minimize(loss)

    return train_op

  def __init__(self, config):
    """Initializes the model.

    Args:
      config: A model configuration object of type Config
    """
    self.config = config
    self.load_data(self.config.f_lab)
    self.add_placeholders()
    self.true_model()
    self.initial_weights()
    self.pred, self.W, self.pool_weights,self.activations, self.mod_effects, self.biases= self.add_model(self.x)
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)
    self.pred_fit = met.prediction_fit(self.y, self.pred)

def write_data(snr_values, data, f_lab):
  filename = f_lab + '_data2'

  with open(filename, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    w.writerow(
     ['snr','step','train_loss','train_fit','test_loss','test_fit','model_fit'])
    for i in range(len(snr_values)):
      snr = snr_values[i]
      for row in data[i]:
        w.writerow([snr] + row)

if __name__ == "__main__":
  f_lab = 'poissrelu'
  snr_values = [None]
  data = []

  for snr in snr_values:
    config = Config(f_lab, snr)
    with tf.Graph().as_default():
      model = ShallowModel(config)
      sess = tf.Session()
      init = tf.global_variables_initializer()
      sess.run(init)
      snr_data = model.fit(sess)

    data.append(snr_data)
  write_data(snr_values, data, f_lab)
