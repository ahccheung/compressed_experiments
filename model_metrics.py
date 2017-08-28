import csv
import numpy as np
import tensorflow as tf


def prediction_fit(y, pred):
  """Calculates fit of predicted values given true values y.
  Returns 1 - |y- pred|^2/|y|^2
  """
  diff_norm = tf.norm(y - pred)
  y_norm = tf.norm(y)
  fit = 1 - diff_norm/y_norm
  fit = tf.cond(tf.is_nan(fit), lambda: 1 - diff_norm, lambda: fit)
  return fit
