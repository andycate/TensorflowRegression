import numpy as np
import tensorflow as tf
import data_aggregation as da

data = da.Data()

x = tf.placeholder(tf.float32, shape=[784, None])
W = tf.Variable(tf.zeros([1, 784]))
b = tf.Variable(tf.zeros([1]))

y = tf.sigmoid(tf.matmul(W, x) + b)
y_ = tf.placeholder(tf.float32, shape=[1, None])
cross_entropy = tf.reduce_mean(-(y_ * tf.log(y)) - ((1 - y_) * (1 - y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = data.next_batch()
    print(batch_xs.transpose().shape)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys.reshape(1, 100)})