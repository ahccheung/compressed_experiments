import csv
import sys
import numpy as np
import tensorflow as tf
import model_metrics as met
import itertools
from model_relu import ReluModel

class Config(object):
  """Holds model hyperparams and data information.
  """
  batch_size = 100
  rank = 2
  order = 3
  max_epochs = 300
  lda1 = 0.01
  lda2 = 1

  def __init__(self, f_lab, snr, lr):
    self.random_init = (snr == None)
    if snr != None:
      self.add_noise = (np.isfinite(snr))
    else: self.add_noise = False
    self.f_lab = f_lab
    self.snr = snr
    self.lr = lr

def write_data(data, f_lab):
  filename = f_lab + '_data_order'
  count = 0
  with open(filename, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    w.writerow(
     ['ind','snr','lr','step','train_loss','train_fit','test_loss','test_fit','model_fit', 'redundancy'])
    for i in range(len(data)):
      for row in data[i]:
        w.writerow([count] + row)
      count +=1

if __name__ == "__main__":

  """Wrapper code for training and evaluating a model. Edit snr_values and
  lr_values for different signal-to-noise ratios and learning rates. SNR of None
  means completely random initialization."""

  f_lab = sys.argv[1]
  snr_values = [None]
  lr_values = [0.1]
  num_trials = 1
  data = []
  params = itertools.product(snr_values, lr_values)
  for (snr, lr) in params:
    for i in range( num_trials):
      config = Config(f_lab, snr, lr)
      model = ReluModel(config)
      sess = tf.Session()
      init = tf.global_variables_initializer()
      sess.run(init)
      model_data = model.fit(sess)
      data.append(model_data)
  write_data(data, f_lab)
