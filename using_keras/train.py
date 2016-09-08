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

#pred = model(x)
# cDNI1 belongs to Layer2, so it accepts Layer1's output and emmits grad_y of Layer1
cDNI1 = Sequential()
cDNI1.add(Dense(1024, input_dim=256+10))
cDNI1.add(BatchNormalization(mode=2))
cDNI1.add(Activation('relu'))
cDNI1.add(Dense(1024))
cDNI1.add(BatchNormalization(mode=2))
cDNI1.add(Activation('relu'))
cDNI1.add(Dense(256, weights=[np.zeros(shape=[1024, 256])], bias=False))
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
cDNI3 = Sequential()
cDNI3.add(Dense(1024, input_dim=256+10))
cDNI3.add(BatchNormalization(mode=2))
cDNI3.add(Activation('relu'))
cDNI3.add(Dense(1024))
cDNI3.add(BatchNormalization(mode=2))
cDNI3.add(Activation('relu'))
cDNI3.add(Dense(256, weights=[np.zeros(shape=[1024, 256])], bias=False))

y_l1 = layer1(x)
p_gy_l1 = cDNI1(K.concatenate((y_l1, labels), axis=1))
grad_trainable_weights_l1 = tf.gradients(y_l1, layer1.trainable_weights, grad_ys=p_gy_l1)
update_l1 = []
for weight, grad in zip(layer1.trainable_weights, grad_trainable_weights_l1):
    update_l1.append(tf.assign(weight, weight-lr*grad))

x_l2 = y_l1
y_l2 = layer2(x_l2)
p_gy_l2 = cDNI2(K.concatenate((y_l2, labels), axis=1))
gy_l1 = tf.gradients(y_l2, y_l1, grad_ys=p_gy_l2)[0]
loss_dni1 = K.mean(K.sum((p_gy_l1-gy_l1)**2, 1))
grad_trainable_weights_dni1 = tf.gradients(loss_dni1, cDNI1.trainable_weights)
with tf.control_dependencies(update_l1):
    update_dni1 = []
    for weight, grad in zip(cDNI1.trainable_weights, grad_trainable_weights_dni1):
        update_dni1.append(tf.assign(weight, weight-lr*grad))
    grad_trainable_weights_l2 = tf.gradients(y_l2, layer2.trainable_weights, grad_ys=p_gy_l2)
    update_l2 = []
    for weight, grad in zip(layer2.trainable_weights, grad_trainable_weights_l2):
        update_l2.append(tf.assign(weight, weight-lr*grad))

    x_l3 = y_l2
    y_l3 = layer3(x_l3)
    p_gy_l3 = cDNI3(K.concatenate((y_l3, labels), axis=1))
    gy_l2 = tf.gradients(y_l3, y_l2, grad_ys=p_gy_l3)[0]
    loss_dni2 = K.mean(K.sum((p_gy_l2-gy_l2)**2, 1))
    grad_trainable_weights_dni2 = tf.gradients(loss_dni2, cDNI2.trainable_weights)
    with tf.control_dependencies(update_dni1+update_l2):
        update_dni2 = []
        for weight, grad in zip(cDNI2.trainable_weights, grad_trainable_weights_dni2):
            update_dni2.append(tf.assign(weight, weight-lr*grad))
        grad_trainable_weights_l3 = tf.gradients(y_l3, layer3.trainable_weights, grad_ys=p_gy_l3)
        update_l3 = []
        for weight, grad in zip(layer3.trainable_weights, grad_trainable_weights_l3):
            update_l3.append(tf.assign(weight, weight-lr*grad))

        x_l4 = y_l3
        y_l4 = layer4(x_l4)
        loss = K.mean(categorical_crossentropy(labels, y_l4))
        gy_l3 = tf.gradients(loss, y_l3)[0]
        loss_dni3 = K.mean(K.sum((p_gy_l3-gy_l3)**2, 1))
        grad_trainable_weights_dni3 = tf.gradients(loss_dni3, cDNI3.trainable_weights)
        with tf.control_dependencies(update_dni2+update_l3):
            update_dni3 = []
            for weight, grad in zip(cDNI3.trainable_weights, grad_trainable_weights_dni3):
                update_dni3.append(tf.assign(weight, weight-lr*grad))
            grad_trainable_weights_l4 = tf.gradients(loss, layer4.trainable_weights)
            update_l4 = []
            for weight, grad in zip(layer4.trainable_weights, grad_trainable_weights_l4):
                update_l4.append(tf.assign(weight, weight-lr*grad))
updates = update_l1+update_l2+update_l3+update_l4+update_dni1+update_dni2+update_dni3+layer1.updates+layer2.updates+layer3.updates+layer4.updates+cDNI1.updates+cDNI2.updates+cDNI3.updates

acc = accuracy(labels, y_l4)



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





