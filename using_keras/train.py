import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as K
from keras.layers import BatchNormalization, Dense
from keras.models import Sequential
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

sess = tf.Session()
K.set_session(sess)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

model = Sequential()
model.add(Dense(256, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

pred = model(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, pred))
acc = accuracy(labels, pred)

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_op = tf.train.AdamOptimizer().minimize(loss)
with sess.as_default():
    sess.run(tf.initialize_all_variables())
    for i in range(5000):
        batch = mnist_data.train.next_batch(256)
        train_op.run(feed_dict={x: batch[0], labels: batch[1]})
        if i%100 == 0:
            print acc.eval(feed_dict={x: mnist_data.test.images, labels: mnist_data.test.labels})





