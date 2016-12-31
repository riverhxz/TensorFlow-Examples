#coding=utf-8
'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.

Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
import numpy as np

from tensorflow.python.ops import rnn, rnn_cell

# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.context = np.random.randint(0,99,size=(n_samples,3,1))
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

n_hidden = 128 # hidden layer num of features
n_embeding = n_hidden
num_words = 3000 # nums of words
num_id,num_cate,num_local = (100,100,100) # id，类别，地域
seq_len = 20
id_type = "int32"

# Network Parameters
seq_max_len = 20 # Sequence max length
# n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

id_type = "int32"

input_id = tf.placeholder(id_type, [None, 1], name="input_id")
input_cate = tf.placeholder(id_type, [None, 1], name="input_cate")
input_local = tf.placeholder(id_type, [None, 1], name="input_local")

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
    return tf.matmul(x, W) + b

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

word_embedings = {
    'words': tf.Variable(tf.random_normal([num_words, n_hidden]))
}
def get_embed(embed, x, name):
    id_embeding = tf.nn.embedding_lookup(embed,x, name='embed_' + name)
    id_embeding = tf.reshape(id_embeding, shape=(-1, n_embeding), name='reshape_embed_' + name)
    return id_embeding

embeds = [get_embed(v, input_xs[k], k) for k,v in embedings.iteritems()]
concat = tf.concat(1, embeds)

h_z = fc(output_dim=n_embeding, x=concat, name='id_embed_fc')

# tf Graph input
input_x = tf.placeholder(id_type, [None, seq_max_len], name="input_word")
y = tf.placeholder("float32", [None], name="label")
# A placeholder for indicating each sequence length
input_seqlen = tf.placeholder(tf.int32, [None], name="input_seq_len")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases, init_state):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 1])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, seq_max_len, x)

    # Define a lstm cell with tensorflow
    gru_cell = rnn_cell.GRUCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.nn.rnn(gru_cell, x, dtype=tf.float32,initial_state=init_state,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

input_word_embed = tf.nn.embedding_lookup(word_embedings['words'], input_x, name='embed_words')
pred = dynamicRNN(input_word_embed, input_seqlen, weights, biases, init_state=h_z)
pred = tf.nn.sigmoid(pred, name="y_hat")
# Define loss and optimizer
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

number_record = 10000
context = np.random.random_integers(0, 99, (number_record, 3, 1))
Y = np.random.randint(0,1,size=(number_record)).astype('float32')
SEQ = np.random.randint(2, num_words - 1, (number_record, seq_len))
SEQ_LENGTHS = np.random.randint(5, seq_len - 1, (number_record))
index_in_epoch = 0

def gen_z_batch(batch_size=32):
    start = index_in_epoch
    end = min(number_record, start + batch_size)
    if end == number_record:
        assert batch_size < number_record
        _index_in_epoch = 0

    return context[start:end], SEQ[start:end], SEQ_LENGTHS[start:end], Y[start:end]
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        b_ctx, b_seq, b_seq_len, b_y  = gen_z_batch(batch_size=32)
        # Run optimization op (backprop)
        feed_dict = {input_id:b_ctx[:,0], input_cate:b_ctx[:,1] , input_local:b_ctx[:,2] , input_x:b_seq, y:b_y , input_seqlen:b_seq_len}
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
    #
    # # Calculate accuracy
    # test_data = testset.data
    # test_label = testset.labels
    # test_seqlen = testset.seqlen
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                   seqlen: test_seqlen}))
