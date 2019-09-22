# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:41:40 2019

@author: LABA
"""

from keras.preprocessing import sequence
from keras.datasets import imdb
from matplotlib import pyplot as plt
import pandas as pd

from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize
import nltk
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout




class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        print('out_dim' , output_dim)
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        #inputs.shape = (batch_size, time_steps, seq_len)
        print('input_shape' , input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        print('kernel', self.kernel)
        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0]) #paper上Q值
        WK = K.dot(x, self.kernel[1]) #paper上K值
        WV = K.dot(x, self.kernel[2]) #paper上V值

        print("WQ.shape",WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)


        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64**0.5)

        QK = K.softmax(QK)

        print("QK.shape",QK.shape)

        V = K.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)



data = pd.read_csv('SPAM text message 20170820 - Data.csv')
message =list(data['Message'])
message_list = []
input_set = set()
input_dict = dict()
for m in message:
    word = nltk.word_tokenize(m)
    message_list.append(word)
    for w in word:
        if w not in input_set:
            input_set.add(w)
   
for i, v in enumerate(input_set):
    input_dict[v] = i

save = []
convert_data = []
maxlen = 0
for ml in message_list:
    save = []
    if maxlen < len(ml): maxlen = len(ml)
    for m in ml:
        if m in input_dict:
            save.append(input_dict[m])
    convert_data.append(save)
    
train = sequence.pad_sequences(convert_data, maxlen, padding='post')
train_x = train[:2000]
test_x = train[2000:]

label = LabelEncoder()
classification = label.fit_transform(list(data['Category']))
label = to_categorical(classification)

train_y = label[:2000]
test_y = label[2000:]


S_inputs = Input(shape=(maxlen,), dtype='int32')

embeddings = Embedding(len(input_set), 200)(S_inputs)


O_seq = Self_Attention(200)(embeddings)


O_seq = GlobalAveragePooling1D()(O_seq)


outputs = Dense(2, activation='softmax')(O_seq)


model = Model(inputs=S_inputs, outputs=outputs)

print(model.summary())
opt = Adam(lr=0.0002,decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss,

             optimizer=opt,

             metrics=['accuracy'])
#
#%%
print('Train...')

h = model.fit(train_x, train_y,

         batch_size=32,

         epochs=5,

         validation_split=0.2)

plt.plot(h.history["loss"],label="train_loss")
plt.plot(h.history["val_loss"],label="val_loss")
plt.plot(h.history["acc"],label="train_acc")
plt.plot(h.history["val_acc"],label="val_acc")
plt.legend()
plt.show()





