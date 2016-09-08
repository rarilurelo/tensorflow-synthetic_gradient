import tensorflow as tf
import numpy as np

class Adam(object):
    def __init__(self, lr, beta_1, beta_2, epsilon, params, sess):
        self.iterations = tf.Variable(0.)
        self.lr = tf.Variable(lr)
        self.beta_1 = tf.Variable(beta_1)
        self.beta_2 = tf.Variable(beta_2)
        self.epsilon = epsilon
        self.ms = [tf.Variable(tf.zeros_like(param)) for param in params]
        self.vs = [tf.Variable(tf.zeros_like(param)) for param in params]
        sess.run(tf.initialize_variables([self.iterations, self.lr, self.beta_1, self.beta_2]+self.ms+self.vs))

    def update(self, params, gparams):
        self.updates = [tf.assign_add(self.iterations, 1)]
        t = self.iterations+1.
        lr_t = self.lr * tf.sqrt(1. - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t))

        for p, g, m, v in zip(params, gparams, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            p_t = p - lr_t * m_t / (tf.sqrt(v_t) + self.epsilon)

            self.updates.append(tf.assign(m, m_t))
            self.updates.append(tf.assign(v, v_t))

            new_p = p_t
            self.updates.append(tf.assign(p, new_p))
        return self.updates

