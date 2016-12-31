#coding=utf-8
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
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
##const
logs_path = '/tmp/tensorflow_logs/example'

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 32
display_step = 10

# Network Parameters
# n_input = 28 # MNIST data input (img shape: 28*28)
# n_steps = 28 # timesteps
# n_hidden = 128 # hidden layer num of features
# n_classes = 10 # MNIST total classes (0-9 digits)

n_hidden = 128 # hidden layer num of features
n_embeding = n_hidden
num_words = 3000 # nums of words
num_id,num_cate,num_local = (100,100,100) # id，类别，地域
seq_len = 20
id_type = "int32"

input_id = tf.placeholder(id_type, [None, 1], name="input_id")
input_cate = tf.placeholder(id_type, [None, 1], name="input_cate")
input_local = tf.placeholder(id_type, [None, 1], name="input_local")
input_word = tf.placeholder(id_type, [None, seq_len], name="input_local")

label_y = tf.placeholder(id_type, [None, seq_len], name="input_local")

input_xs = {
    'ids': input_id,
    'cates': input_cate,
    'locals': input_local
}
# Define weights
embedings = {
    'ids': tf.Variable(tf.random_normal([num_id, n_hidden])),
    'cates': tf.Variable(tf.random_normal([num_cate, n_hidden])),
    'locals': tf.Variable(tf.random_normal([num_local, n_hidden]))
}
word_embeding = {
    'word_input': tf.Variable(tf.random_normal([num_words, n_hidden]))
    # 'word_output': tf.Variable(tf.random_normal([n_hidden, num_words])),
    # 'word_output_bias': tf.Variable(tf.random_normal(num_words))

}


# sampled softmax_loss
num_samples = 50
w = tf.get_variable("proj_w", [n_hidden, num_words])
w_t = tf.transpose(w)
b = tf.get_variable("proj_b", [num_words])


def full_decode(inputs):
    input_tensors = tf.pack(inputs,1)
    input_tensors = tf.reshape(input_tensors,[-1,n_hidden])
    z = tf.matmul(input_tensors, w) + b
    softmax_z = tf.nn.softmax(z, name='greedy_decoder')
    y_hat = tf.argmax(softmax_z ,1)
    return y_hat

def sampled_loss(inputs, labels, num_samples, num_words):
    labels = tf.reshape(labels, [-1, 1])
    input_tensors = tf.pack(inputs,1)
    input_tensors = tf.reshape(input_tensors,[-1,n_hidden])
    return tf.nn.sampled_softmax_loss(w_t, b, input_tensors, labels, num_samples,
                                      num_words)
####### test for embeding#########
np.random.seed(47)

def get_embed(embed, x, name):
    id_embeding = tf.nn.embedding_lookup(embed,x, name='embed_' + name)
    id_embeding = tf.reshape(id_embeding, shape=(-1, n_embeding), name='reshape_embed_' + name)
    return id_embeding

embeds = [get_embed(v, input_xs[k], k) for k,v in embedings.iteritems()]
concat = tf.concat(1, embeds)
params={}
def fc(output_dim, x, name=None):
    '''
    TODO +comment
    :param output_dim:
    :param x:
    :return:
    '''
    input_dim = None
    if type(output_dim) != int:
        raise TypeError('expect type of output_dim is int, found %s' % type(output_dim))

    if type(x) != tf.Tensor:
        raise TypeError('x expected Tensor, found %s' % type(x))
    if name is None:
        name = x.name + "_fc"
    dims = x.get_shape().dims

    if len(dims) == 2:
        input_dim = dims[1].value
    elif len(dims) == 1:
        input_dim = dims[0].value
    else:
        raise ValueError('dim of x expected lt 3, found  %s' % len(dims))

    W = tf.Variable(tf.random_uniform([input_dim, output_dim],0,1), name=name + "_W")
    b = tf.Variable(tf.random_uniform([output_dim],0,1), name=name + "_b")
    params.setdefault(W.name, W)
    params.setdefault(b.name, b)
    return tf.matmul(x, W) + b

h_z = fc(output_dim=n_embeding, x=concat, name='id_embed_fc')

def RNN(x, init_state):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_hidden)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_hidden)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_hidden)
    x = tf.reshape(x, [-1, n_hidden])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    x = tf.split(0, seq_len, x)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn_cell.GRUCell(n_hidden, forget_bias=1.0)
    gru_cell = rnn_cell.GRUCell(n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(gru_cell, x, dtype=tf.float32, initial_state=init_state)

    # Linear activation, using rnn inner loop last output
    return outputs

x=tf.nn.embedding_lookup(word_embeding['word_input'], input_word, name='input_word_embed')
ret_seq = RNN(x, init_state=h_z)

seq_decoded = full_decode(ret_seq)
softmax_loss = sampled_loss(ret_seq, label_y, num_samples, num_words)
cost = tf.reduce_mean(softmax_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

#data
number_record=1000
context = np.random.random_integers(0, 99, (number_record, 3, 1))
Y = np.random.randint(2, num_words - 1, (number_record, seq_len - 1), dtype="int32")
DI_Y = np.concatenate((np.ones(shape=(number_record,1), dtype="int32"), Y), axis=1)
DO_Y = np.concatenate((Y,np.zeros(shape=(number_record,1), dtype="int32")), axis=1)
index_in_epoch = 0

def gen_z_batch(batch_size=32):
    start = index_in_epoch
    end = min(number_record, start + batch_size)
    if end == number_record:
        assert batch_size < number_record
        _index_in_epoch = 0

    return context[start:end], DI_Y[start:end], DO_Y[start:end]
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    # summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    # summary_writer.add_summary(summary, 1)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        ctx, yi, yo = gen_z_batch(batch_size)
        feed_dict = {input_id: ctx[:, 0], input_cate: ctx[:, 1], input_local: ctx[:, 2], input_word:yi, label_y:yo}
        # Run optimization op (backprop)
        sess.run([optimizer], feed_dict=feed_dict)

        if step % display_step == 0:
            # Calculate batch loss
            loss, seq = sess.run([cost, seq_decoded], feed_dict=feed_dict)

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
            print(seq[0:19].tolist())
            print(yo[0,:].tolist())
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict=feed_dict))
