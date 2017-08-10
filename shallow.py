from __future__ import division

import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf

batch_size = 100
flab = 'shallow'
order = 4
# You may adjust the max_epochs to ensure convergence.
max_epochs = 500
# You may adjust this learning rate to ensure convergence.
lr = 0.01
lda = 10
log_dir = os.path.join(os.getcwd(), 'Logs')

def load_data(flab):

  x_train = np.loadtxt(flab + '_trainx', delimiter=',', dtype='float32')
  y_train = np.loadtxt(flab + '_trainy', delimiter=',', dtype ='float32')

  x_test = np.loadtxt(flab + '_testx', delimiter=',', dtype = 'float32')
  y_test = np.loadtxt(flab + '_testy', delimiter=',', dtype = 'float32')

  return x_train, y_train, x_test, y_test

def get_placeholders(input_size, output_size):

  x = tf.placeholder(tf.float32,[None, input_size])
  y = tf.placeholder(tf.float32,[None, output_size])

  return x, y

def inference(input_data, input_size, output_size):
  """Returns prediction and variables from the input data.
  """

  with tf.name_scope('hidden'):

    W = tf.Variable(np.random.exponential(size=(input_size,order)).astype('float32'), name='W')
    activity = tf.Variable(tf.random_normal([1, output_size]), name='act')
    sig = tf.sigmoid(tf.matmul(input_data, W))
    p = tf.expand_dims(tf.cumprod(sig)[:,-1], 1)
    pred  = tf.matmul(p, activity)

  return pred, W, activity

def training(loss):
  """Return training op.
  """
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.AdamOptimizer(lr)

  return optimizer.minimize(loss)

def prediction_fit(y, pred):
  """Returns goodness of fit of prediction.
  """
  diff_norm = tf.norm(y - pred)**2
  y_norm = tf.norm(y)**2
  return 1 - diff_norm/y_norm

def get_loss(y, pred, W):
  """Adds ops for loss to the computational graph.
  Args:
    pred: A tensor of shape (batch_size, output_size)
  Returns:
    loss: A 0-d tensor (scalar) output
  """
  return tf.reduce_mean(tf.norm(y - pred, axis=1)**2)

def create_feed_dict(x_placeholder, y_placeholder, input_data, output_data):
  """Creates the feed_dict for training the given step.
  Args:
    input_data: A batch of input data.
    output_data: A batch of label data.
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  feed_dict = {x_placeholder: input_data, y_placeholder: output_data}

  return feed_dict

def get_batches(batch_size, input_data, output_data=None):
  """"Separates data X,Y into batches of length batch_size.
  Args:
    input_data: A batch of input data.
    output_data: A batch of label data.
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  batches = []
  count = 0

  while len(input_data[count:]) > 0:
    increment = min(batch_size, len(input_data[count:]))
    if np.any(output_data):
      output_batch = output_data[count:count + increment]
    else:
      output_batch = np.array([])
    batch = (input_data[count:count + increment], output_batch)
    count += increment
    batches.append(batch)

  return batches

def main():
  """Create and train model."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():

    x_train, y_train, x_test, y_test = load_data(flab)
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    x, y = get_placeholders(input_size, output_size)
    pred, W, activity = inference(x, input_size, output_size)
    loss = get_loss(y, pred)
    train_op = training(loss)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(init)
    print(sess.run(W))

    batches = get_batches(batch_size, x_train, y_train)
    # Start the training loop.
    test_fd = create_feed_dict(x, y, x_test, y_test)
    train_fd = create_feed_dict(x, y, x_train, y_train)
    for step in xrange(max_epochs):
      start_time = time.time()
      feed_dicts= [create_feed_dict(x,y,batch[0],batch[1]) for batch in batches]
      for i in range(len(batches)):
        # Run one step of the model.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dicts[i])
      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 50 == 0:
        test_loss = sess.run(loss, feed_dict = test_fd)
        train_loss = sess.run(loss, feed_dict = train_fd)
        # Print status to stdout.
        print('Step %d: train loss = %.2f (%.3f sec)'
              % (step, train_loss, duration))
        print('Step %d: test loss = %.2f' % (step, test_loss))
        print(sess.run(W))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=train_fd)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 500 == 0 or (step + 1) == max_epochs:
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

    final_train_loss = sess.run(loss, feed_dict = train_fd)
    final_test_loss = sess.run(loss, feed_dict = test_fd)
    train_fit = sess.run(prediction_fit(y, pred), feed_dict = train_fd)
    test_fit = sess.run(prediction_fit(y, pred), feed_dict = test_fd)

    print('Final train loss: %.2f' %(final_train_loss))
    print('Final train fit: %.2f' %(train_fit))
    print('Final test loss: %.2f' %(final_test_loss))
    print('Final test fit: %.2f' %(test_fit))

if __name__ == "__main__":
  main()
