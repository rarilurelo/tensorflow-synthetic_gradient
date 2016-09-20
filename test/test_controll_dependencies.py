import tensorflow as tf

a = tf.Variable([13])
b = tf.Variable([4])

update_a = tf.assign(a, b)
with tf.control_dependencies([update_a]):
    update_b = tf.assign(b, a)

updates = tf.group(update_a, update_b)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(a)
    print sess.run(b)
    sess.run(updates)
    print sess.run(a)
    print sess.run(b)


