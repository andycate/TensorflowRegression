import numpy as np
import tensorflow as tf
import data_aggregation as da

data = da.Data()

x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.sigmoid(tf.matmul(W, x) + b)
y_ = tf.placeholder(tf.float32, shape=[None, 1])
cross_entropy = tf.reduce_mean(-(y_ * tf.log(y)) - ((1 - y_) * (1 - y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})