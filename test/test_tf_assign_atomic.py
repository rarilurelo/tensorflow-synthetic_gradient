import tensorflow as tf

a = tf.Variable([1.0])
b = tf.Variable([5.0])

#update_a = tf.assign(a, b)
#update_b = tf.assign(b, a)
update_a = tf.assign(a, b, use_locking=True)
c = a
update_b = tf.assign(b, c, use_locking=True)

updates = [[update_a], update_b]
#updates = tf.tuple([update_b], control_inputs=[update_a])
#tf.with_dependencies()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(a)
    print sess.run(b)
    #sess.run(update_a)
    #sess.run(update_b)
    sess.run(updates)
    print sess.run(a)
    print sess.run(b)
