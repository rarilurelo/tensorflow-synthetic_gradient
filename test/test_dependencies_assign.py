import tensorflow as tf

a = tf.placeholder(dtype=tf.float32, shape=[1])

b = tf.Variable([10.])

d = tf.Variable([-1.])
e = tf.Variable([100.])
c = a*b
#update_b = tf.assign(b, b-0.1*c)
rr = b*1
print rr
with tf.control_dependencies([rr]):
    update_b = tf.assign(b, [0])
    with tf.control_dependencies([update_b]):
        update_e = tf.assign(e, d*c)
#update_e = tf.assign(e, d*c)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print sess.run(rr)
print sess.run([update_b, update_e, rr], {a: [2]})
print sess.run(b)
print sess.run(e)
print sess.run(rr)

