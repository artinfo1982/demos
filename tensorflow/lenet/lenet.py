# coding=utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# 定义日志的输出阈值
old_v = tf.logging.get_verbosity()
# 日志级别可以为: DEBUG, INFO, WARN, ERROR, FATAL
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

import matplotlib.pyplot as plt

# 存放训练过程中accuracy和loss的数组
list_accuracy = []
list_loss = []

# 总的迭代次数
niter = 1000
# 画图采样间隔
interval = 100

# one_hot表示用非0即1的数组保存图片表示的数值
# 比如一个图片上写的是0，内存中不是直接存一个0，而是存一个数组[1,0,0,0,0,0,0,0,0,0]，一个图片上面写的是1，那么保存的就是[0,1,0,0,0,0,0,0,0,0]
mnist = input_data.read_data_sets('.', one_hot=True)
tf.logging.set_verbosity(old_v)

# tf.InteractiveSession()的好处在于将会话成为默认会话，后续的run、eval不需要再指定会话
# 而tf.Session()就需要后续run、eval传入session=sess的参数来指定具体的会话
sess = tf.InteractiveSession()

# 输入，float32，shape=[图片张数不固定，大小都是28x28]
x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
# 输出，float32，shape=[图片张数不固定，10（数字0-9的概率）]
# y = wx + b
y = tf.placeholder(tf.float32, shape=[None, 10])
# 将输入调整为4维，-1表示忽略图片的张数，最后的1表示单通道
x_image = tf.reshape(x, [-1, 28, 28, 1])


'''
使用到的函数解释
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)，生成截断的正态分布
tf.Variable(initializer, name)，创建一个变量，并初始化，不填name为匿名
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)
    tensorflow官方给出的建议是，在GPU训练时采用NCHW，在CPU推导时采用NHWC。详见https://www.tensorflow.org/guide/performance/overview?hl=zh-cn
    strides是由4个int型构成的向量，表示每个维度的步长，顺序和data_format中的描述保持一致
    padding=SAME/VALID
    如果使用'SAME'：
        out_height = ceil(float(in_height) / float(strides[1]))，ceil表示向上取整
        out_width = ceil(float(in_width) / float(strides[2]))
    如果使用'VALID'：
        out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
tf.nn.relu(features, name=None)
tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
tf.reshape(tensor, shape, name=None)
tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
    求矩阵点积
    transpose：将矩阵转置
    adjoint：是否共轭矩阵
    is_sparse：是否是稀疏矩阵
tf.nn.softmax(logits, axis=None, name=None, dim=None)
    计算公式：
    softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
tf.reduce_sum(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
    按照维度求和，如果不输入维度信息，将整个矩阵元素求和
tf.log(x, name=None)，计算自然对数，e为底
tf.argmax(input, axis=None, name=None, dimension=None, output_type=tf.int64)
    返回最大值元素的索引，如果不输入维度信息，在整个矩阵中返回最大值的索引
tf.equal(x, y, name=None)，判定x和y是否相等，x和y的类型必须一致，如果是向量或者矩阵，必须里面每个对应位置的元素都相等才返回True
tf.cast(x, dtype, name=None)，将x的每个元素都转换类型dtype
tf.reduce_mean(input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None, keep_dims=None)
    按照维度求平均值，如果不输入维度信息，将整个矩阵元素求平均值
'''

# 第一级：conv+relu+max_pool
w1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1))
b1 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[6]))
conv1 = tf.nn.conv2d(input=x_image, filter=w1, strides=[
                     1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False, data_format='NHWC')
relu1 = tf.nn.relu(conv1 + b1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='SAME')

# 第二级：conv+relu+max_pool
w2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1))
b2 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]))
conv2 = tf.nn.conv2d(input=pool1, filter=w2, strides=[
                     1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False, data_format='NHWC')
relu2 = tf.nn.relu(conv2 + b2)
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[
                       1, 2, 2, 1], padding='SAME')
flat1 = tf.reshape(pool2, [-1, 7 * 7 * 16])

# 第三级：3层全连接
w3 = tf.Variable(tf.truncated_normal(
    shape=[7 * 7 * 16, 120], mean=0, stddev=0.1))
b3 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[120]))
fc1 = tf.nn.relu(tf.matmul(flat1, w3) + b3)

w4 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0, stddev=0.1))
b4 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[84]))
fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)

w5 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0, stddev=0.1))
b5 = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[10]))
output = tf.nn.relu(tf.matmul(fc2, w5) + b5, name='output')

'''
各层输出维度信息如下：
w1 (5, 5, 1, 6)
b1 (6,)
conv1 (?, 28, 28, 6)
relu1 (?, 28, 28, 6)
pool1 (?, 14, 14, 6)
w2 (5, 5, 6, 16)
b2 (16,)
conv2 (?, 14, 14, 16)
relu2 (?, 14, 14, 16)
pool2 (?, 7, 7, 16)
flat1 (?, 784)
w3 (784, 120)
b3 (120,)
fc1 (?, 120)
w4 (120, 84)
b4 (84,)
fc2 (?, 84)
w5 (84, 10)
b5 (10,)
output (?, 10)
'''

# 定义交叉熵损失函数，J = -sum(yi * log(pi))，i为游标
cross_entropy = -tf.reduce_sum(y * tf.log(output))
'''
Adam随机过程优化器，使损失函数最小化
算法详见论文《Adam: A Method for Stochastic Optimization》，https://arxiv.org/abs/1412.6980
Class AdamOptimizer
__init__(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    use_locking=False,
    name='Adam'
)
minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
优化器有很多种，例如：
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
'''
optimizer = tf.train.AdamOptimizer(
    learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cross_entropy)
# 通过argmax分别计算输出，label的概率最大的索引号
# 再判断输出softmax向量最大值和label的是否相等，来判断预测是否准确，返回boll类型
correct_prediction = tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1))
# 求整个结果矩阵的平均值，结果矩阵中元素的值，不是1就是0，1表示识别成功
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

print('train...')
for i in range(niter):
    batch_x, batch_y = mnist.train.next_batch(50)
    _, loss_val = sess.run([optimizer, cross_entropy],
                           feed_dict={x: batch_x, y: batch_y})
    if i % interval == 0:
        # eval()也是启动计算的一种方式。eval()只能用于tf.Tensor类对象，也就是有输出的Operation。
        # 对于没有输出的Operation，可以用.run()或者Session.run()。Session.run()没有这个限制。
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
        # 将accuracy和loss压入相应的数组
        list_accuracy.append(train_accuracy)
        list_loss.append(loss_val)
        print('step %d, train_accuracy: %.2f, loss: %.2f' %
              (i, train_accuracy, loss_val))

# 保存权值为pb文件
constant_graph = graph_util.convert_variables_to_constants(
    sess=sess, input_graph_def=sess.graph_def, output_node_names=['output'])
with tf.gfile.GFile("lenet.pb", mode='wb') as f:
    f.write(constant_graph.SerializeToString())

print('test...')
print('accuracy: %.2f' %
      (accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))

# 画图，可视化训练过程中的accuracy和loss
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(interval * np.arange(len(list_loss)), list_loss,
         color='b', label='loss', linestyle='-')
ax2.plot(interval * np.arange(len(list_accuracy)), list_accuracy,
         color='r', label='accuracy', linestyle='-')
ax1.set_xlabel('iteration')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.savefig('lenet.png')
