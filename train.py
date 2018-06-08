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

def test_error():
    correct_prediction = tf.equal(tf.round(y), tf.round(y_))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy, feed_dict={x: data.test_images, y_: data.test_labels.reshape(1, data.test_labels.shape[0])})

print("before: ", test_error())

for _ in range(100):
    batch_xs, batch_ys = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys.reshape(1, 100)})

print("after: ", test_error())