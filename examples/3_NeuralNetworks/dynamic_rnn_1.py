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
from myworkspace.mlp_loaddata_2 import load_data
from myworkspace.test import cal_auc
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
                 max_value=1000, n_hidden=64, num_id=100):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.id = np.random.randint(2, num_id, (n_samples, 3, 1))
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
        batch_id = (self.id[self.batch_id:min(self.batch_id +
                                               batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen, batch_id

## data
class SampleContainter(object):
    def __init__(self):
        self.data = load_data()
        self.pos = 0
        self.cate = np.expand_dims(self.data["cate"].value, axis=2)
        self.cookie = np.expand_dims(self.data["cookie"].value, axis=2)
        self.local = np.expand_dims(self.data["local"].value, axis=2)
        self.seq = self.data["word"].value
        self.seq_len = self.data["word_len"].value
        x = self.data["y"].value
        self.y = np.zeros((x.shape[0],2))
        self.y[:,0] = x
        self.y[:,1] = 1 - x

        self.config = {}
        self.config["cate_num"] = self.cate.max()
        self.config["cookie_num"] = self.cookie.max()
        self.config["local_num"] = self.local.max()
        self.config["word_num"] = self.seq.max()

        self.config["max_seq_len"] = self.seq.shape[1]
        self.config["number_sample"] = self.seq.shape[0]


    def next_batch(self, batch_size=32):
        if(self.pos == self.config["number_sample"]):
            self.pos = 0
        start = self.pos
        end = min(self.pos + batch_size, len(self.data["cate"]))
        self.pos = end
        return self.cate[start:end], \
               self.cookie[start:end], \
               self.local[start:end],\
                self.seq[start:end], \
            self.seq_len[start:end], \
            self.y[start:end]

    def next_slice(self, batch_size=32):
        if(self.pos == self.config["number_sample"]):
            self.pos = 0
        start = self.pos
        end = min(self.pos + batch_size, len(self.data["cate"]))
        return start, end


train = SampleContainter()

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

# Network Parameters
seq_max_len = train.config["max_seq_len"]  # Sequence max length
n_hidden = 8 # hidden layer num of features
n_classes = 2 # linear sequence or not

#mysetting
num_id = train.config["cookie_num"] + 1
num_cate = train.config["cate_num"] + 1
num_local = train.config["local_num"] + 1
num_embeding = n_hidden
num_word = train.config["word_num"] + 1

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
id_type = "int32"

input_id = tf.placeholder(id_type, [None, 1], name="input_id")
input_cate = tf.placeholder(id_type, [None, 1], name="input_cate")
input_local = tf.placeholder(id_type, [None, 1], name="input_local")

input_word_embeding = tf.Variable(tf.random_normal([num_word, n_hidden]))
embedings = {
    'id': tf.Variable(tf.random_normal([num_id, n_hidden])),
    'cate': tf.Variable(tf.random_normal([num_cate, n_hidden])),
    'local': tf.Variable(tf.random_normal([num_local, n_hidden]))
}

input_xs = {
    'id': input_id,
    'cate': input_cate,
    'local': input_local
}

def get_embed(embed, x, name):
    id_embeding = tf.nn.embedding_lookup(embed,x, name='embed_' + name)
    id_embeding = tf.reshape(id_embeding, shape=(-1, num_embeding), name='reshape_embed_' + name)
    return id_embeding


# trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
# testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input

x = tf.placeholder(tf.int32, [None, seq_max_len])
y = tf.placeholder("float", [None, n_classes])


# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

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

    # x = tf.transpose(x, [1, 0, 2])
    # # Reshaping to (n_steps*batch_size, n_input)
    # x = tf.reshape(x, [-1, n_input])
    # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.split(0, seq_max_len, x)

    # Define a lstm cell with tensorflow
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    # outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32,
    #                             sequence_length=seqlen, initial_state=init_state)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                sequence_length=seqlen, initial_state=init_state)
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    # outputs = tf.pack(outputs)
    # outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    output_dim = tf.shape(outputs)[2]
    # Start indices for each sample
    # 验证seq数据正确
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, output_dim]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

# z1 = get_embed(embedings["id"], z, name='id_embed')

embeds = [get_embed(v, input_xs[k], k) for k,v in embedings.iteritems()]
concat = tf.concat(1, embeds)

z_fc = fc(n_hidden,concat, name='z_fc')
emb_x = tf.nn.embedding_lookup(input_word_embeding,x, name='word_embed')

pred = dynamicRNN(emb_x, seqlen, weights, biases, z_fc)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#auc
xxx = np.zeros((train.config["number_sample"],2))
xxx[:,1] = train.data["y"].value
step_per_epoch = train.config["number_sample"] / batch_size + 1


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        start, end = train.next_slice(batch_size)
        b_cate, b_id, b_local, b_seq, b_seq_len, b_y = train.next_batch(batch_size)
        # Run optimization op (backprop)
        feed_dict = {x: b_seq, y: b_y,
                     input_id: b_id, input_local: b_local, input_cate: b_cate,
                     seqlen: b_seq_len}
        predict_val, _ = sess.run([pred,optimizer], feed_dict=feed_dict)
        xxx[start:end,0] = predict_val[:,1]
        # if step % display_step == 0:
        #     # Calculate batch accuracy
        #     acc = sess.run(accuracy, feed_dict=feed_dict)
        #     # Calculate batch loss
        #     loss = sess.run(cost, feed_dict=feed_dict)
        #     print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
        #           "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #           "{:.5f}".format(acc))
        if step % display_step == 0:
            print ("epoch {:3f}%".format((step % step_per_epoch)*100. / step_per_epoch))
        if step % step_per_epoch == 0:
            auc = cal_auc(xxx)
            print ("AUC:{:.6f}".format(arc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
