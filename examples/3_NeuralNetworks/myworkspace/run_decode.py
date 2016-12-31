# -*- coding: utf-8 -*-
import numpy as np
import gc
import sys
import keras.backend as K
from keras.backend.tensorflow_backend import *
from keras.engine import Model
from keras.models import Sequential
from keras.engine import merge
from keras.layers import LeakyReLU
from mlp_loaddata import load_data,sample2words
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Input,Dropout,Merge,Embedding,Flatten,BatchNormalization
from keras.layers.wrappers import TimeDistributed
from seq2seq.cells import LSTMDecoderCell
from recurrentshop import LSTMCell, RecurrentContainer
from keras import initializations, regularizers, activations


def pre_process():
    MAX_SEQUENCE_LENGTH = 20
    f_data_head = '/Users/hehehe/PycharmProjects/keras/myworkspace/testdata/sample.head.pkl'

    train, test = load_data(path=f_data_head,nb_words=10000,maxlen=20)
    X,Y = train
    X = np.asanyarray(X,dtype='int32')
    Y = pad_sequences(Y, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    Y = np.asanyarray(Y,dtype='int32')

    config = (X[:,0].max() + 1, X[:,1].max()+ 1,X[:,2].max()+ 1, Y.max()+ 1) # number of id,local,cate,word
    Y = np.reshape(Y,(Y.shape[0],Y.shape[1],1))
    gc.collect()
    return X,Y,config


'''
input order [id_input, local_input, cate_input]

'''

def create_model(cate_num=10, local_num=10, id_num=20, \
                 hidden_dim=128, seq_len=20, word_num = 10000, inner_broadcast_state=True, broadcast_state=True, \
                 encoder_dim=64, dropout=0.5,depth=1, output_length=20, peek=False, teacher_force=True, batch_size = 20,\
                 ** kwargs):

    if type(depth) == int:
        depth = [depth, depth]
    if 'batch_input_shape' in kwargs:
        shape = kwargs['batch_input_shape']
        del kwargs['batch_input_shape']
    elif 'input_shape' in kwargs:
        shape = (None,) + tuple(kwargs['input_shape'])
        del kwargs['input_shape']
    elif 'input_dim' in kwargs:
        if 'input_length' in kwargs:
            shape = (None, kwargs['input_length'], kwargs['input_dim'])
            del kwargs['input_length']
        else:
            shape = (None, None, kwargs['input_dim'])
    if 'unroll' in kwargs:
        unroll = kwargs['unroll']
        del kwargs['unroll']
    else:
        unroll = False
    if 'stateful' in kwargs:
        stateful = kwargs['stateful']
        del kwargs['stateful']
    else:
        stateful = False

    hid_rep_dim=64
    output_dim = word_num
    def withMask(layer):
        layer._keras_history[0].supports_masking = True



    cate_input = Input(batch_shape=(shape[0],1), name="cate")
    withMask(cate_input)
    local_input = Input(batch_shape=(shape[0],1),  name='local')
    withMask(local_input)
    id_input = Input(batch_shape=(shape[0],1), name='id')
    withMask(id_input)

    def flatEmb(input, input_dim, output_dim, input_lenghth=1):
        emb = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_lenghth)(input)
        return Flatten()(emb)

    cate_rep = flatEmb(input=cate_input,input_dim=cate_num, output_dim=hid_rep_dim)
    local_rep = flatEmb(local_input,local_num,hid_rep_dim)
    id_rep = flatEmb(id_input, id_num, hid_rep_dim)
    words_rep

    fc1 = LeakyReLU()(Dense(64)(cate_rep))
    fc2 = LeakyReLU()(Dense(64)(local_rep))
    fc3 = LeakyReLU()(Dense(64)(id_rep))

    params = merge([fc1, fc2, fc3], mode='concat', concat_axis=1)
    params = Dense(output_dim)(params)
    decoder_input =  LeakyReLU()(params)
    decoder_input = BatchNormalization()(decoder_input)
    decoder_input = Dropout(dropout)(decoder_input)

    em


    decoder = RecurrentContainer(readout='readout_only', state_sync=inner_broadcast_state, \
                                 output_length=output_length, unroll=unroll, stateful=stateful, decode=True, \
                                 input_length=hidden_dim,return_sequences=True)
    for _ in range(1,depth[1]):
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim, batch_input_shape=(shape[0], output_dim),**kwargs))

    out_lstm = LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim, batch_input_shape=(shape[0], output_dim),**kwargs)
    out_lstm.activation=activations.get('softmax')
    decoder.add(out_lstm)
    states = [decoder_input, None]
    inputs = [id_input, local_input, cate_input]

    # if teacher_force:
    #     truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
    #     truth_tensor._keras_history[0].supports_masking = True
    #     inputs += [truth_tensor]
    decoded_seq = decoder(
        {'input': decoder_input,
         'states': states,
         'initial_readout': decoder_input})

    model = Model(inputs, decoded_seq)

    return model
    #model.compile(optimizer=optimizer, loss='msle')

def train_model():

    x, y, config = pre_process() #x.shape = batch,3; y.shape = batch, outputlen, 1
    id, local, cate, word = config
    # sz = 1000
    # dim = 100
    word_output_dim = word
    output_len = 20
    # x = np.random.random_integers(0,dim - 1,(sz,3))
    # y = np.random.random_integers(0,word_output_dim - 1,(sz,output_len,1))
    model = create_model(cate_num=cate,local_num=local,id_num=id,word_num=word_output_dim,output_length=output_len, batch_size=32,unroll=True,input_shape=(output_len,),depth=2)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    #res = model.predict([x[:,0],x[:,1],x[:,2]])
    model.fit([x[:,0],x[:,1],x[:,2]],y,batch_size=32, nb_epoch=100)
    #print "result shape: %d %d" % (len(res), len(res[0]))
    print res

def load_model():
    x, y, config = pre_process() #x.shape = batch,3; y.shape = batch, outputlen, 1
    id, local, cate, word = config
    # sz = 1000
    # dim = 100
    word_output_dim = word
    output_len = 20
    # x = np.random.random_integers(0,dim - 1,(sz,3))
    # y = np.random.random_integers(0,word_output_dim - 1,(sz,output_len,1))
    model = create_model(cate_num=cate,local_num=local,id_num=id,word_num=word_output_dim,output_length=output_len, batch_size=32,unroll=True,input_shape=(output_len,),depth=3)

    filePath='/private/tmp/model3/model.90.hdf5'
    model.load_weights(filepath=filePath)
    sz = 50
    x=x[:sz]
    y=y[:sz,:,0]
    top = 5
    h = model.predict([x[:,0],x[:,1],x[:,2]])
    #h_y = np.argmax(h,axis=2)
    h_y =np.argpartition(h,-top,axis=2)[:,:,-top:]
    words_h_y = sample2words(h_y)
    words_y = sample2words(y)
    for i in range(sz):
        print words_h_y[i], words_y[i]



if __name__ == '__main__':
    #train_model()
    #pre_process()
    load_model()
    print 'done'