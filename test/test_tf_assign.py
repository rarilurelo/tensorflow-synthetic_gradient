import tensorflow as tf

p = tf.placeholder(dtype=tf.float32, shape=[1])
a = tf.Variable([3.0])
M_1 = tf.Variable([5.0])
b = tf.Variable([7.0])
M_2 = tf.Variable([11.0])

a_0 = p
a_1 = p*a
ga = a_1*M_1
update_a = tf.assign(a, ga, use_locking=True)
#update_a = tf.assign(a, ga, use_locking=False)
a_2 = a_1*b
gb = a_2*M_2
gM_1 = gb*b

update_M_1 = tf.assign(M_1, gM_1, use_locking=True)
#update_M_1 = tf.assign(M_1, gM_1, use_locking=False)

update_b = tf.assign(b, gb, use_locking=True)
#update_b = tf.assign(b, gb, use_locking=False)

updates = tf.group(update_a, update_M_1, update_b)
#updates = tf.group(update_a, update_b, update_M_1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(a)
    print sess.run(M_1)
    print sess.run(b)
    print sess.run(M_2)

    print sess.run(updates, {p: [1.0]})

    print sess.run(a)
    print sess.run(M_1)
    print sess.run(b)
    print sess.run(M_2)


