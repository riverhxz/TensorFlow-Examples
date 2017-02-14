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
from myworkspace.mlp_loaddata_2 import SampleTransform
from myworkspace.test import cal_auc2,cal_auc1

class Datasets(object):
    def __init__(self,  container, start, end):

        # self.data = SampleTransform().load_data("/home/hehehe/data/sample.10m.h5")
        # self.usage = int(usage * self.data["word"].value.shape[0])
        self.start = start
        self.end = end
        self.pos = start
        self.data = container
        self.sample_num = end - start


    def next_batch(self, batch_size=32):
        if self.pos == self.end:
            self.pos = self.start

        idx_start = self.pos
        idx_end = min(self.pos + batch_size, self.end)
        self.pos = idx_end
        return self.data.cate[idx_start:idx_end], \
               self.data.cookie[idx_start:idx_end], \
               self.data.local[idx_start:idx_end], \
               self.data.seq[idx_start:idx_end], \
               self.data.seq_len[idx_start:idx_end], \
               self.data.y[idx_start:idx_end]


    def next_slice(self, batch_size=32):
        if self.pos == self.end:
            self.pos = self.start

        idx_start = self.pos
        idx_end = min(self.pos + batch_size, self.end)
        return idx_start, idx_end


class SampleContainter(object):
    def __init__(self,  fn, train_percent=1):

        data = SampleTransform().load_data(fn)

        self.config = {}
        self.config["cate_num"] = data["cate"].value.max()
        self.config["cookie_num"] = data["cookie"].value.max()
        self.config["local_num"] = data["local"].value.max()
        self.config["word_num"] = data["word"].value.max()
        word_shape = data["word"].value.shape
        self.config["max_seq_len"] = word_shape[1]
        self.config["number_sample"] = word_shape[0]

        split = int(train_percent * self.config["number_sample"])
        self.split = split
        self.cate = np.expand_dims(data["cate"].value, axis=2)
        self.cookie = np.expand_dims(data["cookie"].value, axis=2)
        self.local = np.expand_dims(data["local"].value, axis=2)
        self.seq = data["word"].value
        self.seq_len = data["word_len"].value
        #类型flot16
        x = data["y"].value
        self.y = np.zeros((x.shape[0],2))
        self.y[:,0] = x
        self.y[:,1] = 1 - x

        self.train_dataset = Datasets(self, 0, split)
        self.test_dataset = Datasets(self, split, self.config["number_sample"])

data_wrapper = SampleContainter('/home/hehehe/data/sample.25m.h5',train_percent=0.8)

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 900000000
batch_size = 1024*32
display_step = 100
epsilon = 0.0000001
# Network Parameters
seq_max_len = data_wrapper.config["max_seq_len"]  # Sequence max length
num_embeding = 16
n_hidden = num_embeding * 4 # hidden layer num of features
n_classes = 2 # linear sequence or not

#mysetting
# num_id = 654718
# num_cate = 884
# num_local = 12106
# num_word = 5701

num_id = data_wrapper.config["cookie_num"] + 1
num_cate = data_wrapper.config["cate_num"] + 1
num_local = data_wrapper.config["local_num"] + 1
num_word = data_wrapper.config["word_num"] + 1


print("num_id:{:d} num_cate:{:d} num_local:{:d} word_num:{:d}".format(num_id, num_cate, num_local, num_word))
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

    W = tf.Variable(tf.random_uniform([input_dim, output_dim],0,1, dtype=float_type), name=name + "_W")
    b = tf.Variable(tf.random_uniform([output_dim],0,1, dtype=float_type), name=name + "_b")
    return tf.matmul(x, W) + b
id_type = "int32"
float_type = tf.float32
input_id = tf.placeholder(id_type, [None, 1], name="input_id")
input_cate = tf.placeholder(id_type, [None, 1], name="input_cate")
input_local = tf.placeholder(id_type, [None, 1], name="input_local")

input_word_embeding = tf.Variable(tf.random_normal([num_word, num_embeding], dtype=float_type))
embedings = {
    'id': tf.Variable(tf.random_normal([num_id, num_embeding * 2], dtype=float_type)),
    'cate': tf.Variable(tf.random_normal([num_cate, num_embeding], dtype=float_type)),
    'local': tf.Variable(tf.random_normal([num_local, num_embeding], dtype=float_type))
}

input_xs = {
    'id': input_id,
    'cate': input_cate,
    'local': input_local
}

def get_embed(embed, x, name):
    id_embeding = tf.nn.embedding_lookup(embed,x, name='embed_' + name)
    id_embeding = tf.reshape(id_embeding, shape=(-1, embed.get_shape()[1].value), name='reshape_embed_' + name)
    return id_embeding


# tf Graph input

x = tf.placeholder(id_type, [None, seq_max_len])
y = tf.placeholder(float_type, [None, n_classes])


# A placeholder for indicating each sequence length
seqlen = tf.placeholder(id_type, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], dtype=float_type))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes], dtype=float_type))
}

def batch_norm(x):
    mean, var = tf.nn.moments(x,[0])
    return tf.nn.batch_normalization(x, mean, var,None,None,epsilon, name='bn')


def dynamicRNN(x, seqlen, weights, biases, init_state):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)

    # Define a lstm cell with tensorflow
    cell = tf.nn.rnn_cell.GRUCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    # outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32,
    #                             sequence_length=seqlen, initial_state=init_state)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=float_type,
                                sequence_length=seqlen, initial_state=init_state, parallel_iterations=128)
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
    # 验证seqlen > max_seqlen
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, output_dim]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

# z1 = get_embed(embedings["id"], z, name='id_embed')

embeds = [get_embed(v, input_xs[k], k) for k,v in embedings.iteritems()]
concat = tf.concat(1, embeds)

z_fc = fc(n_hidden,concat, name='z_fc')
tf.maximum()
b_fc = batch_norm(z_fc)
emb_x = tf.nn.embedding_lookup(input_word_embeding,x, name='word_embed')

z_output = dynamicRNN(emb_x, seqlen, weights, biases, b_fc)
pred = tf.nn.softmax(z_output, name='pred')
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(z_output, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)



# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
model_path ='model/xxx.model'
def train(data_wrapper):
    # Launch the graph
    # Initializing the variables
    init = tf.initialize_all_variables()
    train_set = data_wrapper.train_dataset
    #auc
    xxx = np.zeros((train_set.sample_num, 2))
    xxx[:,1] = data_wrapper.y[:data_wrapper.split, 0]
    step_per_epoch = train_set.sample_num / batch_size + 1

    sum_cost = 0.

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        epoch = 1
        best_loss = 1.
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            start, end = train_set.next_slice(batch_size)
            b_cate, b_id, b_local, b_seq, b_seq_len, b_y = train_set.next_batch(batch_size)
            # Run optimization op (backprop)
            feed_dict = {x: b_seq, y: b_y,
                         input_id: b_id, input_local: b_local, input_cate: b_cate,
                         seqlen: b_seq_len}
            cost_val, predict_val, _ = sess.run([cost, pred, optimizer], feed_dict=feed_dict)
            sum_cost += cost_val * (end - start)

            xxx[start:end,0] = predict_val[:,0]


            if step % display_step == 0:
                print (" {:2.0f}%".format((step % step_per_epoch)*100. / step_per_epoch),end=' '),
            if step % (step_per_epoch) == 0:
                auc = cal_auc2(xxx)
                avg_loss = sum_cost / train_set.sample_num
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = saver.save(sess, '{:s}.{:d}'.format(model_path,epoch))

                print ("=====AUC:{:.6f}====".format(auc) + ' loss:{:.6f}'.format(avg_loss))
                # print("=====AUC:{:.6f}====".format(cal_auc1(xxx)) + ' loss:{:.6f}'.format(sum_cost / train.config["number_sample"]))
                epoch += 1
                sum_cost = 0.
            step += 1
        print("Optimization Finished!")

def test(data_wrapper):
    test_set = data_wrapper.test_dataset
    # auc
    xxx = np.zeros((test_set.sample_num, 2))
    xxx[:, 1] = data_wrapper.y[data_wrapper.split:, 0]
    step_per_epoch = test_set.sample_num / batch_size + 1

    sum_cost = 0.
    with tf.Session() as sess:
        saver.restore(sess,model_path + '.21')
        step = 1
        epoch = 1
        best_loss = 1.
        # Keep training until reach max iterations
        while step * batch_size < test_set.sample_num or step == 1:
            start, end = test_set.next_slice(batch_size)
            b_cate, b_id, b_local, b_seq, b_seq_len, b_y = test_set.next_batch(batch_size)
            # Run optimization op (backprop)
            feed_dict = {x: b_seq, y: b_y,
                         input_id: b_id, input_local: b_local, input_cate: b_cate,
                         seqlen: b_seq_len}
            cost_val, predict_val = sess.run([cost, pred], feed_dict=feed_dict)
            sum_cost += cost_val * (end - start + 1)
            xxx[start - test_set.start:end - test_set.start , 0] = predict_val[:, 0]
            if end == test_set.end:
                break
            if step % display_step == 0:
                print(" {:2.0f}%".format((step % step_per_epoch) * 100. / step_per_epoch), end=' '),

            step += 1

        auc = cal_auc2(xxx)
        avg_loss = sum_cost / test_set.sample_num

        print("=====AUC:{:.6f}====".format(auc) + ' loss:{:.6f}'.format(avg_loss))


test(data_wrapper)