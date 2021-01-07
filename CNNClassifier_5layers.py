# Deep learning CNN
import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
tf.set_random_seed(777)  # reproducibility
data = np.loadtxt("/root/jiwon/cc2.csv", delimiter=',')

print(type(data))
np.random.shuffle(data)
print(data.shape)
print(data[:10])

# data = data[:3500]

train_num = int(len(data) * 0.8)
print(train_num)

# feature, label
x_train, t_train = data[:train_num, :-1], tf.one_hot(data[:train_num, -1], 2)  # one_hot은 tensor이기 때문에 나중에 배열로 바꿔줘야 한다.
x_test, t_test = data[train_num:, :-1], tf.one_hot(data[train_num:, -1], 2)

# hyper parameters
learning_rate = 0.007
training_epochs = 25
batch_size = 100


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.decay = 0.99
        self.training = tf.placeholder(tf.bool, name='training')

        self._build_net()  # 이거 마지막에

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 5625])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 75, 75, 1])
            self.Y = tf.placeholder(tf.float32, [None, 2])

            # CONVOLUTION 1
            W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D:0", shape=(?, 75, 75, 32), dtype=float32)
            Tensor("Relu:0", shape=(?, 75, 75, 32), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 38, 38, 32), dtype=float32)
            Tensor("dropout/mul:0", shape=(?, 38, 38, 32), dtype=float32)
            '''

            # CONVOLUTION 2
            W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D_1:0", shape=(?, 38, 38, 64), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 38, 38, 64), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 19, 19, 64), dtype=float32)
            Tensor("dropout_1/mul:0", shape=(?, 19, 19, 64), dtype=float32)
            '''

            # CONVOLUTION 3
            W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            #             L3 = tf.contrib.layers.batch_norm(L3, decay=self.decay, scale=True,
            #                                               is_training=self.training, updates_collections=None)  # 배치정규화
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3_flat = tf.reshape(L3, [-1, 128 * 10 * 10])
            '''
            Tensor("Conv2D_2:0", shape=(?, 19, 19, 128), dtype=float32)
            Tensor("Relu_2:0", shape=(?, 19, 19, 128), dtype=float32)
            Tensor("MaxPool_2:0", shape=(1, 10, 10, 128), dtype=float32)
            Tensor("dropout_2/mul:0", shape=(?, 10, 10, 128), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 12800), dtype=float32)
            '''

            # AFFINE 1
            W4 = tf.get_variable("W4", shape=[128 * 10 * 10, 512], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([512]))
            L4 = tf.matmul(L3_flat, W4) + b4
            #             L4 = tf.contrib.layers.batch_norm(L4, decay=self.decay, scale=True,
            #                                               is_training=self.training, updates_collections=None)  # 배치정규화
            L4 = tf.nn.relu(L4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_3:0", shape=(?, 512), dtype=float32)
            Tensor("dropout_3/mul:0", shape=(?, 512), dtype=float32)
            '''

            # AFFINE 2
            W5 = tf.get_variable("W5", shape=[512, 2], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([2]))
            self.logits = tf.matmul(L4, W5) + b5
            '''
            Tensor("add_1:0", shape=(?, 2), dtype=float32)
            '''

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test.eval(session=self.sess),
                                                       self.keep_prob: keep_prob})
        # one_hot의 tensor 데이터를 array 형태로 바꾼다.

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data.eval(session=self.sess),
                                        self.keep_prob: keep_prob})
        # one_hot의 tensor 데이터를 array 형태로 바꾼다.


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(data) / batch_size)  # 총 배치의 개수
    total_size = len(data)
    print(total_size, total_batch)

    for i in range(0, total_size, batch_size):
        print(i)
        batch_xs, batch_ys = x_train[i:i + batch_size], t_train[i:i + batch_size]
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy_test:', m1.get_accuracy(x_test, t_test))
# print('Accuracy_train:', m1.get_accuracy(x_train, t_train))
