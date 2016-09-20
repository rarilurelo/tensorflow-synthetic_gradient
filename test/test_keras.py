from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf

sess = tf.Session()
K.set_session(sess)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
model = Sequential()
model.add(Dense(256, input_dim=784))
model.add(BatchNormalization(mode=2))
model.add(Dense(256))
model.add(Dense(256))

hoge = Sequential()
hoge.add(Dense(256, input_dim=256))

model.add(hoge)
for w in model.trainable_weights:
    print sess.run(w)
a = model.trainable_weights
y = model(x)
for w in model.trainable_weights:
    print sess.run(w)

print model.trainable_weights
print model.layers[0].trainable_weights[0].get_shape()

print model.layers[1].trainable_weights

print Activation('relu').name
print Dense(256, input_dim=888).get_weights()
print model.get_weights()[0].shape

def res():
    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Dense(256))
    return model


