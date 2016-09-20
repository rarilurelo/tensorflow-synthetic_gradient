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


# forward
y_l1 = layer1(x)
x_l2 = y_l1
y_l2 = layer2(x_l2)
x_l3 = y_l2
y_l3 = layer3(x_l3)
x_l4 = y_l3
y_l4 = layer4(x_l4)
loss = K.mean(categorical_crossentropy(labels, y_l4))


# backward form loss
from_loss_grad_trainable_weights_l4 = tf.gradients(loss, layer4.trainable_weights)
from_loss_grad_trainable_weights_l3 = tf.gradients(loss, layer3.trainable_weights)
from_loss_grad_trainable_weights_l2 = tf.gradients(loss, layer2.trainable_weights)
from_loss_grad_trainable_weights_l1 = tf.gradients(loss, layer1.trainable_weights)

# backward from upward layer
grad_trainable_weights_l4 = tf.gradients(loss, layer4.trainable_weights)
gx_l4 = tf.gradients(loss, x_l4)
grad_trainable_weights_l3 = tf.gradients(y_l3, layer3.trainable_weights, grad_ys=gx_l4)
gx_l3 = tf.gradients(y_l3, x_l3, gx_l4)
grad_trainable_weights_l2 = tf.gradients(y_l2, layer2.trainable_weights, gx_l3)
gx_l2 = tf.gradients(y_l2, x_l2, gx_l3)
grad_trainable_weights_l1 = tf.gradients(y_l1, layer1.trainable_weights, gx_l2)

# update
updates = []
for trainable_weights, grad_trainable_weights in zip([layer1.trainable_weights, layer2.trainable_weights, layer3.trainable_weights, layer4.trainable_weights],
                                                     [grad_trainable_weights_l1, grad_trainable_weights_l2, grad_trainable_weights_l3, grad_trainable_weights_l4]):
    for weight, grad in zip(trainable_weights, grad_trainable_weights):
        updates.append(tf.assign(weight, weight-lr*grad))
acc = accuracy(labels, y_l4)

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

with sess.as_default():
    sess.run(tf.initialize_all_variables())
    for i in range(50000):
        batch = mnist_data.train.next_batch(256)
        sess.run(updates, feed_dict={x: batch[0], labels: batch[1], K.learning_phase(): 1})
        if i%1000 == 0:
            hoge = sess.run(grad_trainable_weights_l1, feed_dict={x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0})
            hoge2 = sess.run(from_loss_grad_trainable_weights_l1, feed_dict={x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0})
            for i, j in zip(hoge, hoge2):
                if (np.logical_not(i-j)).all() :
                    print 'hoge'

            print "epoch: {}".format(256*i//len(mnist_data.train.images))
            print "acc: {}".format(acc.eval(feed_dict={x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0}))
            print "loss: {}".format(loss.eval({x: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0}))






