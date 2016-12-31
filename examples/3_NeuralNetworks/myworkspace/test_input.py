#coding=utf-8
from keras import activations
from keras.layers import Input, Embedding, LSTM, Dense, merge, LeakyReLU,Flatten
from keras.models import Model
import numpy as np

import keras.backend as K
# headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# note that we can name any layer by passing it a "name" argument.
cate_num=10
main_input = Input(shape=(cate_num,), name='main_input')
id_num=10000
id_rep_dim=100

#id_emb = Embedding(input_dim=10000,output_dim=id_rep_dim,input_length=1)(id_input)
#id_flat = Flatten()(id_emb)


from seq2seq.cells import LSTMDecoderCell
from recurrentshop import LSTMCell, RecurrentContainer

def test_trans_embed():
    id_input = Input(shape=(1,), dtype='int32')
    x_input = Input(shape=(1,), dtype='int32')

    def flatEmb(input,input_dim, output_dim, input_lenghth=1):
        emb=Embedding(input_dim=input_dim,output_dim=output_dim,input_length=input_lenghth)(input)
        return Flatten()(emb)

    x_emb  = flatEmb(input=x_input, input_dim=id_num, output_dim=id_rep_dim)
    id_emb = flatEmb(input=id_input, input_dim=id_num, output_dim=id_rep_dim)

    merge_emb = merge([x_emb,id_emb],mode='concat',concat_axis=1)
    # this embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensio)nal vectors.
    x = Dense(output_dim=10)(merge_emb)
    auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(x)
    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    inputs = [id_input, x_input]
    m = Model(inputs, x_emb)
    m.compile(optimizer='sgd',loss='mse')

    data = np.random.random_integers(0,10000,(10,2))
    res = m.predict(x=[data[:,0], data[:,1]], batch_size=1)
    print "result is %s" % len(res), len(res[0])
    print res

def test_word_embed():
    sz = 1000
    word_embed_dim = 100
    word_dim = 100
    word_length = 20

    filepath = '/tmp/testmodel'

    shape = (10,)
    y = np.random.random_integers(0,word_dim - 1,(sz,word_length - 1,1))
    y_output = np.concatenate((y,np.zeros((sz,1,1))), axis=1)
    y_input = np.concatenate((np.zeros((sz,1,1)), y), axis=1)[:,:,0]
    sample_x_context_rep = np.random.random((sz,word_embed_dim))


    x_input = Input(shape=(word_length,),name='decoder_input', dtype='int32')
    x_input._keras_history[0].supports_masking = True

    context_input = Input(shape=(word_embed_dim,),name='context_input')
    context_input._keras_history[0].supports_masking = True
    # 加上这货之后，创建gradient符号链接的时候就出错了...

    mem_input = Dense(word_embed_dim, name='mem_input')(context_input)
    converse_input = Dense(word_embed_dim, name='h_input')(context_input)
    x_emb = Embedding(input_dim=word_dim,output_dim=word_embed_dim, input_length=word_length, name='decoder_embeding')(x_input) # sz * output_len * output_dim

    decoder = RecurrentContainer( state_sync=True, \
                                 output_length=word_dim, unroll=True, stateful=False, \
                                 input_length=word_length, return_sequences=True, name='decoder')
    output_lstm = LSTMDecoderCell(output_dim=word_dim,
                                  hidden_dim=word_embed_dim,
                                  input_dim=word_embed_dim,
                                  input_shape=(word_length, word_embed_dim, ),name='decoder_inbound_1')
    output_lstm.activation = activations.get('softmax')
    decoder.add(
        output_lstm
            )

    dec_seq = decoder(
        {'input': x_emb,
         'states': [context_input, None]}
    )

    m = Model([x_input, context_input], dec_seq)
    m.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy')
    #m.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    m.fit([y_input, sample_x_context_rep], y_output, nb_epoch=100)
    m.save(filepath=filepath)

if __name__ == '__main__':
    test_word_embed()