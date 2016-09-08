from __future__ import division
import tensorflow as tf
from keras.datasets import mnist
from layers import Dense, DenseDNI, Softmax
from sklearn.utils import shuffle
import numpy as np
from optimizers import Adam

in_dim = 784
cat_dim = 10
hid_dim = 256
epochs = 5
batch_size = 100
sess = tf.Session()

layers = [Dense(in_dim, hid_dim, sess),
          DenseDNI(hid_dim, hid_dim, hid_dim, cat_dim, sess),
          DenseDNI(hid_dim, hid_dim, hid_dim, cat_dim, sess),
          DenseDNI(hid_dim, hid_dim, hid_dim, cat_dim, sess),
          DenseDNI(hid_dim, cat_dim, hid_dim, cat_dim, sess)]

x = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
t = tf.placeholder(dtype=tf.float32, shape=[None, cat_dim])

params = []
y = x
for i, layer in enumerate(layers):
    params += layer.get_params()
    y = layer.forward(y)
    if i < len(layers)-1:
        # DNI set predicted gy to layer
        layers[i+1].predict_grad(tf.concat(1, [y, t]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, t))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

last_layer = layers[-1]
gy = tf.gradients(loss, last_layer.y)[0]

gparams = []
for i, layer in enumerate(layers):
    # backpropagate p_gy through one layer
    # set gW, gb, gx
    if i < len(layers)-1:
        layer.backward(layers[i+1].p_gy)
    else:
        layer.backward(gy)
    gparams += layer.gparams

# extracting y and gx from tensorflow's graph to lock layer's update before M's update
gxs = []
ys = []
for layer in layers:
    gxs.append(layer.gx)
    ys.append(layer.y)
gxs.pop(0)
ys.pop()

# update layer
optimizer4layer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, params=params, sess=sess)
update_layer = optimizer4layer.update(params, gparams)

# update M
updates_M = []
for i, layer in enumerate(layers):
    if i < len(layers)-1:
        p_gy = layers[i+1].predict_grad(tf.concat(1, [layer.y_lock, t]))
        t_gy = layers[i+1].gx_lock
        params_dni = layers[i+1].params_dni
        loss_l2 = tf.reduce_mean(tf.reduce_sum((p_gy-t_gy)**2, reduction_indices=[1]))
        gparams_dni = tf.gradients(loss_l2, params_dni)
        optimizer4M = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, params=params_dni, sess=sess)
        update_M = optimizer4M.update(params_dni, gparams_dni)
        updates_M.append(update_M)

gxs_lock = []
ys_lock = []
for layer in layers:
    gxs_lock.append(layer.gx_lock)
    ys_lock.append(layer.y_lock)
gxs_lock.pop(0)
ys_lock.pop()

# training

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train = np.eye(10)[y_train].astype(np.float32)
y_test = np.eye(10)[y_test].astype(np.float32)
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
X_train = X_train/255.0
X_test = X_test/255.0

#sess.run(tf.initialize_all_variables())
for epoch in range(epochs):
    X_train, y_train = shuffle(X_train, y_train)
    for i in range(X_train.shape[0]//batch_size):
        feed_dict = {x: X_train[i*batch_size:(i+1)*batch_size], t: y_train[i*batch_size:(i+1)*batch_size]}
        _gxs = sess.run(gxs, feed_dict)
        _ys = sess.run(ys, feed_dict)
        sess.run(update_layer, feed_dict)

        for key_gx, value_gx, key_y, value_y, update_M in zip(gxs_lock, _gxs, ys_lock, _ys, updates_M):
            feed_dict = {}
            feed_dict[key_gx] = value_gx
            feed_dict[key_y] = value_y
            feed_dict[t] = y_train[i*batch_size:(i+1)*batch_size]
            sess.run(update_M, feed_dict)
    feed_dict = {x: X_test, t: y_test}
    print "epoch: {}, loss: {}, acc: {}".format(epoch, sess.run(loss, feed_dict), sess.run(accuracy, feed_dict))


