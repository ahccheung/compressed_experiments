from __future__ import division
import csv
import numpy as np
import tensorflow as tf

def prediction_fit(y, pred):
  """Calculates fit of predicted values given true values y.
  Returns 1 - |y- pred|/|y|
  """
  diff_norm = tf.norm(y - pred)
  y_norm = tf.norm(y)
  fit = 1 - diff_norm/y_norm
  fit = tf.cond(tf.is_nan(fit), lambda: 1 - diff_norm, lambda: fit)
  return fit

def subset_metrics(pred, true):
  true_reduced = np.apply_along_axis(sum, 1, true)
  pred_reduced = np.apply_along_axis(sum, 1, pred)
  return single_col_metrics(pred_reduced,true_reduced)

def single_col_metrics(pred, true):
  true_pos = np.logical_and(pred!=0, true!=0)
  false_pos = np.logical_and(pred!=0, true==0)
  false_neg = np.logical_and(pred==0, true!=0)
  count_list = [true_pos, false_pos, false_neg]
  counts =[np.add.reduce(x, axis = None, dtype='float') for x in count_list]

  #ratios: true_pos/ (true_pos + false_neg + 1.5*false_pos)
  return counts[0]/(counts[0] + counts[2] + 1.5*counts[1])

def pairwise_redundancies(pred, true):
  J = pred.shape[0]
  pred_pos = get_redundancies(pred)
  pred_neg = np.logical_not(pred_pos)
  true_pos = get_redundancies(true)
  true_neg = np.logical_not(true_pos)
  offset = J*(J-1)/2
  total = J*(J+1)/2
  tp = np.add.reduce(np.logical_and(pred_pos, true_pos), axis = None)
  tn = np.add.reduce(np.logical_and(pred_neg, true_neg), axis = None) - offset
  fp = np.add.reduce(np.logical_and(pred_pos, true_neg), axis = None)
  fn = np.add.reduce(np.logical_and(pred_neg, true_pos), axis = None)
  return [x/total for x in [tp,tn, fp, fn]]

def get_redundancies(x):
  red = lambda col: np.outer(col, col)!=0
  redundancies =np.array([red(col) for col in x.T])
  return np.tril(np.logical_or.reduce(redundancies, axis = 0))

def redundancy_metrics(model, pred, true):
  red_across_ranks = np.array([redundancy_fit(pred[:,l,:], true[:,l,:]) for l in range(model.config.rank)])
  return np.mean(red_across_ranks)

def redundancy_fit(pred, true):
  f = lambda col: single_col_metrics(find_ref(pred, col), col)
  mets  = np.apply_along_axis(f, 0, true)
  return np.mean(mets)

def find_ref(pred, true):

  fp = lambda x: np.logical_and(x!=0, true==0)
  fn = lambda x: np.logical_and(x==0, true!=0)
  f_pos = np.add.reduce(np.apply_along_axis(fp, 0, pred), axis=0, dtype='float')
  f_neg = np.add.reduce(np.apply_along_axis(fn, 0, pred), axis=0, dtype='float')
  score = 1.5*f_pos + f_neg

  return pred[:,np.argmin(score)]

def model_met(sess,model):
  train_fd = model.create_feed_dict(model.x_train, model.y_train)
  test_fd = model.create_feed_dict(model.x_test, model.y_test)

  train_loss, train_fit= sess.run(
           [model.loss, model.pred_fit], feed_dict=train_fd)

  test_loss, test_fit = sess.run(
           [model.loss, model.pred_fit], feed_dict=test_fd)
  redundancy, offset_fit, final_w_fit = model_fit(sess, model)
  mod_fit = (redundancy + offset_fit + final_w_fit) /3

  print 'Loss %f'  %(train_loss)
  print 'Prediction fit %f'  %(train_fit)
  print 'Model fit %f, %f, %f' %(redundancy, offset_fit, final_w_fit)
  #print model.pruned
  #print model.V_true

  return [train_loss, train_fit, test_loss, test_fit, mod_fit, redundancy]

def model_fit(sess, model):

  redundancy = redundancy_metrics(model, model.pruned, model.V_true)

  fit1 = (prediction_fit(model.offset_true, model.offset)\
         + prediction_fit(model.weights_true, model.weights)) /2

  fit2 = (prediction_fit(model.mod_true, model.mod_effects) \
         + prediction_fit(model.bias_true,model.biases)) /2

  return redundancy, sess.run(fit1), sess.run(fit2)
