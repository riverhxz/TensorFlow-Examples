from keras.layers import Input, Embedding, LSTM, Dense, merge, LeakyReLU,Flatten,Lambda,BatchNormalization
from keras.models import Model

import numpy as np

input_dim=4
output_dim=5
hid_dim = 10
size_data=100000
main_input = Input(shape=(input_dim,), name='main_input')
b = LeakyReLU()(Dense(output_dim=hid_dim)(main_input))
b = Dense(output_dim=output_dim,activation='softmax')(main_input)

model = Model(input=main_input, output=b)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

x = np.random.random((size_data,input_dim))
y = np.random.random_integers(0,output_dim - 1,(size_data))


model.fit(x,y)
#result = model.predict(x)
#print result