import tensorflow as tf
import numpy as np

class Dense(object):
    """Linear calculation"""
    def __init__(self, in_dim, out_dim, sess, initial_W=None):
        if initial_W is None:
            #initial_W = np.random.uniform(low=-np.sqrt(6./(in_dim+out_dim)), high=np.sqrt(6./(in_dim+out_dim)), size=[in_dim, out_dim])
            initial_W = tf.truncated_normal(shape=[in_dim, out_dim])
        self.W = tf.Variable(initial_W, dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(shape=[out_dim]), dtype=tf.float32)
        sess.run(tf.initialize_variables([self.W, self.b]))
        # Activation is constraint to only relu
        self.activation = tf.nn.relu
        #self.params = [self.W, self.b]
        self.gx_lock = tf.placeholder(dtype=tf.float32, shape=[None, in_dim])
        self.y_lock = tf.placeholder(dtype=tf.float32, shape=[None, out_dim])
        self.params = [self.W, self.b]

    def get_params(self):
        return [self.W, self.b]

    def forward(self, x):
        self.x = x
        self._y = tf.matmul(self.x, self.W)+self.b
        self.y = self.activation(self._y)
        return self.y

    def backward(self, grad):
        grad = grad*tf.select((self._y > 0), tf.ones_like(self._y), tf.zeros_like(self._y))
        self.gW = tf.matmul(self.x, grad, transpose_a=True)
        self.gb = tf.reduce_sum(grad, reduction_indices=(0))
        self.gx = tf.matmul(grad, self.W, transpose_b=True)
        self.gparams = [self.gW, self.gb]

class DenseDNI(Dense):
    def __init__(self, in_dim, out_dim, hid_dim, cat_dim, sess, initial_W=None):
        super(DenseDNI, self).__init__(in_dim, out_dim, sess)
        self.dni = [Dense(in_dim+cat_dim, hid_dim, sess), Dense(hid_dim, in_dim, sess, tf.zeros(shape=[hid_dim, in_dim]))]

    def predict_grad(self, y):
        self.p_gy = y
        self.params_dni = []
        for layer in self.dni:
            self.p_gy = layer.forward(self.p_gy)
            self.params_dni += layer.params
        return self.p_gy

class Softmax(object):
    """softmax"""
    def __init__(self):
        self.activation = tf.nn.softmax
        self.params = []

    def forward(self, x):
        self.x = x
        return self.activation(x)

    def backward(self, grad):
        return grad*self.activation(self.x)*(1-self.activation(self.x))
