'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
n_embeding = n_hidden


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
z = tf.placeholder("float", [None, n_embeding])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
np.random.seed(47)
no_example = mnist.train.num_examples

z_data = np.random.random((no_example, n_embeding))
index_in_epoch = 0
def gen_z_batch(batch_size=32):

    start = index_in_epoch
    end = min(no_example, start + batch_size)
    if end == no_example:
        assert batch_size < no_example
        _index_in_epoch = 0

    return z_data[start:end]

def fc(output_dim, x):
    input_dim = None
    if type(output_dim) != int:
        raise ValueError('expect type of output_dim is int, found %s' % type(output_dim))

    if type(x) != tf.Tensor:
        raise ValueError('x expected Tensor, found %s' % type(x))

    dims = x.get_shape().dims

    if len(dims) == 2:
        input_dim = dims[1].value
    elif len(dims) == 1:
        input_dim = dims[0].value
    else:
        raise ValueError('dim of x expected lt 3, found  %s' % len(dims))

    W = tf.Variable(tf.zeros([input_dim, output_dim]))
    b = tf.Variable(tf.zeros([output_dim]))
    return tf.matmul(x, W) + b

h_z = fc(output_dim=n_embeding, x=z)
#c_z = fc(output_dim=n_embeding, x=z)

def RNN(x, weights, biases, init_state):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn_cell.GRUCell(n_hidden, forget_bias=1.0)
    gru_cell = rnn_cell.GRUCell(n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(gru_cell, x, dtype=tf.float32, initial_state=init_state)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases, init_state=h_z)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        feed_dict = {x: batch_x, z: gen_z_batch(batch_size), y: batch_y}
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict=feed_dict)
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict=feed_dict)
            # Calculate batch loss
            loss = sess.run(cost, feed_dict=feed_dict)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict=feed_dict))
