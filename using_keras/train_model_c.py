import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Activation
from keras.models import Sequential
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import numpy as np

sess = tf.Session()
K.set_session(sess)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])
lr = 0.0001

# Layer1
layer1 = Sequential()
layer1.add(Dense(256, input_dim=784))
layer1.add(BatchNormalization(mode=2))
layer1.add(Activation('relu'))
# Layer2
layer2 = Sequential()
layer2.add(Dense(256, input_dim=256))
layer2.add(BatchNormalization(mode=2))
layer2.add(Activation('relu'))
# Layer3
layer3 = Sequential()
layer3.add(Dense(256, input_dim=256))
layer3.add(BatchNormalization(mode=2))
layer3.add(Activation('relu'))
# Layer4
layer4 = Sequential()
layer4.add(Dense(10, activation='softmax', input_dim=256))

# cDNI1 belongs to Layer2, so it accepts Layer1's output and emmits grad_y of Layer1
#cDNI1 = Sequential()
#cDNI1.add(Dense(1024, input_dim=256+10))
#cDNI1.add(BatchNormalization(mode=2))
#cDNI1.add(Activation('relu'))
#cDNI1.add(Dense(1024))
#cDNI1.add(BatchNormalization(mode=2))
#cDNI1.add(Activation('relu'))
#cDNI1.add(Dense(256, weights=[np.zeros(shape=[1024, 256])], bias=False))
# cDNI2 belongs Layer3
cDNI2 = Sequential()
cDNI2.add(Dense(1024, input_dim=256+10))
cDNI2.add(BatchNormalization(mode=2))
cDNI2.add(Activation('relu'))
cDNI2.add(Dense(1024))
cDNI2.add(BatchNormalization(mode=2))
cDNI2.add(Activation('relu'))
cDNI2.add(Dense(256, weights=[np.zeros(shape=[1024, 256])], bias=False))
# cDNI3 belongs Layer4
#cDNI3 = Sequential()
#cDNI3.add(Dense(1024, input_dim=256+10))
#cDNI3.add(BatchNormalization(mode=2))
#cDNI3.add(Activation('relu'))
#cDNI3.add(Dense(1024))
#cDNI3.add(BatchNormalization(mode=2))
#cDNI3.add(Activation('relu'))
#cDNI3.add(Dense(256, weights=[np.zeros(shape=[1024, 256])], bias=False))

layer1.add(layer2)
layer_p = layer1
layer3.add(layer4)
layer_n = layer3
y_p = layer_p(x)
p_gy_p = cDNI2(K.concatenate((y_p, labels), axis=1))
grad_trainable_weights_p = tf.gradients(y_p, layer_p.trainable_weights, grad_ys=p_gy_p)

x_n = y_p
y_n = layer_n(x_n)
loss = K.mean(categorical_crossentropy(labels, y_n))
grad_trainable_weights_n = tf.gradients(loss, layer_n.trainable_weights)
gy_p = tf.gradients(loss, y_p)
loss_dni2 = K.mean(K.sum((p_gy_p-gy_p)**2, 1))
grad_trainable_weights_dni2 = tf.gradients(loss_dni2, cDNI2.trainable_weights)

with tf.control_dependencies(grad_trainable_weights_dni2+grad_trainable_weights_p+grad_trainable_weights_n):
    update_n = []
    for weight, grad in zip(layer_n.trainable_weights, grad_trainable_weights_n):
        update_n.append(tf.assign(weight, weight-lr*grad))
    update_dni2 = []
    for weight, grad in zip(cDNI2.trainable_weights, grad_trainable_weights_dni2):
        update_dni2.append(tf.assign(weight, weight-lr*grad))
    update_p = []
    for weight, grad in zip(layer_p.trainable_weights, grad_trainable_weights_p):
        update_p.append(tf.assign(weight, weight-lr*grad))
updates = update_n+update_dni2+update_p

acc = accuracy(labels, y_n)



mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

with sess.as_default():
    sess.run(tf.initialize_all_variables())
    for i in range(500000):
        batch = mnist_data.train.next_batch(256)
        sess.run(updates, feed_dict={x: batch[0], labels: batch[1], K.learning_phase(): 1})
        if i%10 == 0:
            print "epoch: {}".format(256*i//len(mnist_data.train.images))
            print "acc: {}".format(acc.eval(feed_dict={x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0}))
            print "loss: {}".format(loss.eval({x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0}))





