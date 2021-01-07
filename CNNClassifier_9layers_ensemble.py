import tensorflow as tf
import numpy as np
import time

# hyper parameters
learning_rate = 0.001
training_epochs = 23

print('loading....')
# data = np.loadtxt('/root//data//EleRhi//image_data_eleph_rhino.csv', delimiter=',')

# data = np.loadtxt('/root/data/EleRhi/image_data_eleph_rhino.csv', delimiter=',') #jinwoo
batch_size =100
data = np.loadtxt('merge.csv', delimiter=',')

np.random.shuffle(data)
# data = data[:18000]
data[:, : 75*75] = data[:, : 75*75] / 255

print(len(data))
print('loaded!')
train_num = round(int(data.shape[0]) * .8)
print(train_num)
print(train_num)  # 6402
print(data.shape)  # 91 45, 5626
y_tmp = np.zeros([len(data), 2])

for i in range(len(data)):
    label = int(data[i][-1])
    y_tmp[i, label - 1] = 1
y = y_tmp.tolist()
####나누긴 했는데 one-hot이 안됨
# x_train, t_train = data[:train_num, 0:-1], tf.one_hot(data[:train_num,-1],2)
# x_test, t_test =  data[train_num:, 0:-1], tf.one_hot(data[train_num:,-1],2)
x_train, t_train = data[:train_num, 0:-1], y[:train_num]
x_test, t_test = data[train_num:len(data), 0:-1], y[train_num:len(data)]


def BN(input, training, name, scale=True, decay=0.99):
    return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None,
                                        scope=name)

# print(x_train.shape, t_train.shape)
# x_train = tf.cast(x_train, np.array)
# t_train = tf.cast(t_train, np.array)
# t_train = tf.one_hot(t_train, depth=2)
# t_train = tf.reshape(t_train, [-1, 2])
# t_train = tf.reshape(t_train, [-1, 2])
# print('t_train.shape',t_train.shape) # (6402,2)
print('t_train[0]', t_train[0])

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 5625])
            X_img = tf.reshape(self.X, [-1, 75, 75, 1])  # 색상 하나
            self.Y = tf.placeholder(tf.float32, [None, 2])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=None)
            conv1_bn = BN(input=conv1, training=1, name='conv1_bn')
            conv1_bn_rl = tf.nn.relu(conv1_bn, name='conv1_bn_rl')
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv2_bn = BN(input=conv2, training=1, name='conv2_bn')
            conv2_bn_rl = tf.nn.relu(conv2_bn, name='conv2_bn_rl')
            pool2 = tf.layers.max_pooling2d(inputs=conv2_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv3_bn = BN(input=conv3, training=1, name='conv3_bn')
            conv3_bn_rl = tf.nn.relu(conv3_bn, name='conv3_bn_rl')
            pool3 = tf.layers.max_pooling2d(inputs=conv3_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv4_bn = BN(input=conv4, training=1, name='conv4_bn')
            conv4_bn_rl = tf.nn.relu(conv4_bn, name='conv4_bn_rl')
            pool4 = tf.layers.max_pooling2d(inputs=conv4_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=[3, 3],
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     padding="SAME", activation=tf.nn.relu)
            conv5_bn = BN(input=conv5, training=1, name='conv5_bn')
            conv5_bn_rl = tf.nn.relu(conv5_bn, name='conv5_bn_rl')
            pool5 = tf.layers.max_pooling2d(inputs=conv5_bn_rl, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            print(pool5.shape)  # 5*5*256
            # Dense Layer with Relu
            flat = tf.reshape(pool5, [-1, 3 * 3 * 256])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=512, activation=tf.nn.relu)
            dense5 = tf.layers.dense(inputs=dense4,
                                     units=1024, activation=tf.nn.relu)
            dense6 = tf.layers.dense(inputs=dense5,
                                     units=1024, activation=tf.nn.relu)

            # dropout5 = tf.layers.dropout(inputs=dense4,
            #                              rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dense6, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def etrain(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
# with tf.Session() as sess:
#     with tf.device("/gpu:1"):

sess = tf.Session()
models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')
stime = time.time()
# train my model
for epoch in range(training_epochs):

    avg_cost_list = np.zeros(len(models))
    total_batch = int(train_num / batch_size)

    # print('total_batch', total_batch)
    for step in range(0, train_num, batch_size):
        batch_xs, batch_ys = np.array(x_train[step:step + batch_size]), np.array(t_train[step:step + batch_size])
        # print('batch_xs.shape', batch_xs.shape)
        # print('batch_xs.type', batch_xs.type)
        # train each model

        for m_idx, m in enumerate(models):
            c, _ = m.etrain(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
etime = time.time()
print('consumption time : ', round(etime-stime, 6))
print('Learning Finished!')

# Test model and check accuracy
# train_num = len(mnist.test.labels)
print('Test Started!')
print(train_num)

predictions = np.zeros(len(x_test) * 2).reshape(len(x_test), 2)


test_len = len(x_test)
NumOfModel=len(models)

tot = test_len/batch_size
model_accuracy = [0., 0., 0., 0., 0.]
for step in range(0, test_len, batch_size):

    if step + batch_size > test_len:
        a,b = test_len,test_len

        print('over!!')
    else:
        a = step + batch_size
        b = batch_size
    print('a', a)
    print('b', b)
    for m_idx, m in enumerate(models):

        model_result = np.zeros(b, dtype='int32')

        model_accuracy[m_idx] +=  m.get_accuracy(x_test[step:a], t_test[step:a])/tot
        p = m.predict(x_test[step:a])
        # model_result += np.argmax(p,1)
        model_result[:] += np.argmax(p,1)   # 가장 큰거 골라서 예측한 레이블

        for idx, result in enumerate(model_result):
            # print(result)
            predictions[step+idx, result] += 1

for i in range(len(model_accuracy)):
    print('Model '+str(i)+' Accuracy: '+str(model_accuracy[i]))
ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(t_test, 1))  # 같은지 안같은지 불리언으로 출력
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32)) # 숫자로 바꿔서 평균
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

#
# for i  in range(0, total_batch2, batch_size2):
#     acc = 0
#     xt2, tt2 = x_test[i:batch_size], t_test[i:i+batch_size]
#     acc = m1.get_acc(xt2, tt2)
#     total += acc/batch_num
#     print(acc)
# print('acc: ', total)