#coding:utf-8
import tensorflow as tf

# a=tf.constant([[1.0,2.0]])    #1行2列
# b=tf.constant([[3.0],[4.0]])  #2行1列
#
# #计算图
# result=tf.matmul(a,b)
# print(result)
#
# # 会话
# with tf.Session() as sess:
#     print(sess.run(result))

# w=tf.Variable(tf.random_normal([2,3],stddev=2,mean=0,seed=1))
# tf.random_normal()正态分布，[2,3]2x3矩阵，标准差=2，均值为0，随机种子seed=1，没有seed=1，下次产生的随机数和这次不一样
# tf.truncated_normal() 去掉过大偏离点的正态分布
# tf.random_uniform()平均分布

# 初始化全是0的数组
# a=tf.zeros([3,2])
# 初始化全是1的数组
# a=tf.ones([3,2])
# 全定值数组
# a=tf.fill([3,2],6)
# 直接给值
# a=tf.constant([3,2,1])
# # # 会话
# with tf.Session() as sess:
#     print(sess.run(a))
# print(a)

# #######神经网络前向输出过程####################
# 定义输入和参数
# x=tf.constant([[0.7,0.5]])
# x=tf.placeholder(tf.float32,shape=(1,2))    #一组数据，2维
# x=tf.placeholder(tf.float32,shape=(None,2)) # n组数据，2维
# w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
# # 定义前向传播过程
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
#
# #用会话计算结果
# with tf.Session() as sess:
#     # 初始化所有变量
#     init_op=tf.global_variables_initializer()
#     sess.run(init_op)
#     # print("y in tf3_3.py is :\n", sess.run(y)   #如果x输入形式是x=tf.constant([[0.7,0.5]])
#     # print("y in tf3_3.py is :\n",sess.run(y,feed_dict={x:[[0.7,0.5]]}))    #如果x输入形式是x=tf.placeholder(tf.float32,shape=(1,2))
#     print("y in tf3_3.py is :\n",sess.run(y, feed_dict={x: [[0.7, 0.5],[0.2, 0.3],[0.3, 0.4],[0.4, 0.5]]})) # 如果x输入形式是x=tf.placeholder(tf.float32,shape=(None,2))
#     print("w1:\n",sess.run(w1))
#     print("w2:\n", sess.run(w2))
# #######神经网络前向输出过程####################

# #######神经网络反向传播计算####################

# #######神经网络反向传播计算####################