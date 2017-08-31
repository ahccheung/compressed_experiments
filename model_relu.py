import csv
import operator
import numpy as np
from scipy import special
import tensorflow as tf
import model_metrics as met

class ReluModel(object):
  """Implements model for shallow decomposition.
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
      r = csv.reader(csvfile, delimiter=',')
      self.weights_true = np.array(r.next()).astype('float32')
      self.weights_true = self.weights_true.reshape((self.config.rank,1))
      self.V_true = np.empty((self.input_size, self.config.rank, self.config.order))
      for j in range(self.input_size):
        for l in range(self.config.rank):
          self.V_true[j,l,:] = r.next()
      self.V_true = self.V_true.astype('float32')

    with open(self.config.f_lab + 'module_effects', 'rb') as csvfile:
      r = csv.reader(csvfile,delimiter = ',')
      self.mod_true = np.array(r.next()).astype('float32')
      self.mod_true = self.mod_true.reshape((1, self.output_size))
      self.bias_true = np.array(r.next()).astype('float32')
      self.bias_true = self.bias_true.reshape((1, self.output_size))

    self.pool_true = np.zeros(self.config.order)
    self.offset_true = np.loadtxt(self.config.f_lab + 'offsets', delimiter=',', dtype = 'float32')

  def get_variances(self):
    """Get variances of true weights.
    Returns:
      List of variances for V, mod, bias, act, pool, k.
    """

    f = lambda x: np.mean(x)**2
    weights = [self.V_true, self.mod_true, self.bias_true, self.offset_true,self.pool_true, self.weights_true]
    mu_2 = [f(x) for x in weights]
    g = lambda x: np.sqrt(x/self.config.snr)

    return [g(mu) for mu in mu_2]

  def initial_weights(self):
    """Initializes weights for models by adding noise to true weights.
    """

    if self.config.random_init:
      self.V_init = np.random.normal(0, 1, (self.input_size,self.config.rank,self.config.order)).astype('float32')
      self.mod_init = np.random.poisson(1, (1, self.output_size)).astype('float32')
      self.bias_init = np.random.poisson(1, (1, self.output_size)).astype('float32')
      self.offset_init = np.random.poisson(1,(1, self.config.rank)).astype('float32')
      self.pool_init = np.zeros((self.config.rank, self.config.order))
      self.pool_init = (self.pool_init + np.random.normal(0,1, (1, self.config.order))).astype('float32')
      self.weights_init =np.random.randn(self.config.rank, 1).astype('float32')

    elif not self.config.add_noise:
      self.V_init = self.V_true
      self.mod_init = self.mod_true
      self.bias_init = self.bias_true
      self.offset_init = self.offset_true
      self.pool_init = self.pool_true
      self.weights_init = self.weights_true
    else:
      var = self.get_variances()
      self.V_init = (self.V_true + np.random.normal(0, var[0],(self.input_size,self.config.rank,self.config.order))).astype('float32')
      self.mod_init = (self.mod_true + abs(np.random.normal(0, var[1], (1, self.output_size)))).astype('float32')
      self.bias_init = (self.bias_true + abs(np.random.normal(0, var[2], (1, self.output_size)))).astype('float32')
      self.offset_init = (self.offset_true + np.random.normal(0, var[3], (1, self.config.rank))).astype('float32')
      self.pool_init = (self.pool_true + np.random.normal(0, var[4], (1, self.config.order))).astype('float32')
      self.weight_init = (self.weights_true + np.random.normal(0, var[5], (self.config.rank, 1))).astype('float32')

  def add_model(self, input_data):
    """Implements core of model that transforms input_data into predictions.

    Args:
      input_data: A tensor of shape (batch_size, input_size).
    """
    # Hidden
    with tf.name_scope('hidden'):
      self.V = tf.abs(tf.Variable(
        self.V_init, name = 'Conv_weights'))
      conv = tf.tensordot(self.x, self.V, 1)

      self.pool_weights = tf.Variable(self.pool_init, dtype='float32',name = 'Pool_weights')
      pooled = self.pool(conv, self.pool_weights)

      self.offset = tf.Variable(self.offset_init, name = 'Offset')
      relu = tf.nn.relu(pooled - self.offset)
      self.weights = tf.Variable(self.weights_init)
      module = tf.matmul(relu, self.weights)

      self.mod_effects = tf.abs(tf.Variable(
               self.mod_init, name = 'Module_activity'))
      self.biases = tf.abs(tf.Variable(
             self.bias_init, name = 'Biases'))
      self.pred  = tf.matmul(module, self.mod_effects) + self.biases

  def pool(self, conv, pool_weights):
    """Continuous version of product pooling.
    Each input x is transformed to 1 - sigmoid(weight) + sigmoid(weight) * x.
    Eg: If tf.sigmoid(weight) = 0, get x -> 1
        If tf.sigmoid(weight) = 0.5, get x -> 0.5 + 0.5x
        If tf.sigmoid(weight) = 1, get x -> x
    Then transformed inputs are then product pooled.

    Args:
      conv: tensor of length d, representing inner products of x with module
      element weights.
      pool_weights: weights for pooling

    Returns:
      Tensor of product pooled values."""

    pool_vec = tf.sigmoid(pool_weights)
    weighted = 1 - pool_vec + tf.multiply(conv, pool_vec)
    return tf.cumprod(weighted, axis = 2)[:,:,-1]

  def add_loss_op(self, pred):
    """Adds ops for loss to the computational graph.
    Args:
      pred: A tensor of shape (batch_size, output_size)
    """
    reg = self.config.lda1 *self.config.batch_size* tf.norm(self.V, ord = 1) \
          + self.config.lda2*self.config.batch_size* tf.norm(tf.sigmoid(self.pool_weights), ord=1)

    mean = tf.reduce_mean(tf.norm(self.y - pred, axis=1))
    self.loss = reg + mean

  def run_epoch(self, sess, input_data, output_data):
    """Runs an epoch of training.
    Args:
      sess: tf.Session() object
      input_data: np.ndarray of shape (n_samples, input_size)
      output_labels: np.ndarray of shape (n_samples, output_size)
    """

    batches = self.batches(input_data, output_data,self.config.batch_size)
    for (x, y) in batches:
      feed = self.create_feed_dict(x, y)
      _ = sess.run([self.train_op], feed_dict=feed)

  def batches(self, X,  Y, batch_size):
    """"Separates data X,Y into random batches of length batch_size.
    Args:
      X: ndarray of input training data of size (n_samples, input_size)
      Y: ndarray of output training data of size (n_samples, output_size)
      batch_size: integer giving the size of each batch
    """
    batches = []
    count = 0
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    while len(idx[count:]) > 0:
      increment = min(batch_size, len(idx[count:]))
      if np.any(Y):
        Ybatch = Y[idx[count:count + increment]]
      else:
        Ybatch = np.array([])
      batch = (X[idx[count:count + increment]], Ybatch)
      count += increment
      batches.append(batch)

    return batches

  def prune_weights(self, V, pool_weights):
    threshold = 7
    pool_vec =special.expit(pool_weights)
    pool_vec[pool_weights>threshold] = 1
    pool_vec[pool_weights<-threshold] = 0

    V = np.multiply(V, pool_vec)
    pruned = np.empty((self.input_size, self.config.rank, self.config.order))
    for l in range(self.config.rank):
      pruned[:,l,:] = np.array([self.prune_col(col) for col in V[:,l,:].T]).astype('float32').T

    return pruned

  def prune_col( self, col):
    col_max = max(abs(col))
    prune = [10*abs(cell)< col_max and cell!= 0 for cell in col]
    if not np.any(prune):
      return col
    else:
      tmp = np.copy(col)
      tmp[prune] = 0
      return self.prune_col(tmp)

  def fit(self, sess):
    """Fit model on provided data by performing stochastic gradient descent.
    Evaluate model every 100 steps and return performance statistics
    Args:
      sess: tf.Session() object.
    Returns:
      data: list of performance statistics of length (max_epochs/100) + 1 (1 for
      every 100 steps.)
    """
    data = []
    train_fd = self.create_feed_dict(self.x_train, self.y_train)
    test_fd =self.create_feed_dict(self.x_test, self.y_test)

    for i in range(self.config.max_epochs):

      if (i % 100 ==0):
        print 'Step %d:' %(i)
        self.pruned= self.prune_weights(sess.run(self.V), sess.run(self.pool_weights))
        metrics  = met.model_met(sess, self)
        data.append( [self.config.snr, self.config.lr]+ [i] + metrics)
      self.run_epoch(sess, self.x_train, self.y_train)

    metrics = met.model_met(sess, self)
    data.append([self.config.snr, self.config.lr, self.config.max_epochs] + metrics)
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
    self.add_model(self.x)
    self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)
    self.pred_fit = met.prediction_fit(self.y, self.pred)
