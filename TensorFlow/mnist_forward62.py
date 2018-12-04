#coding:utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf

INPUT_NODE=784  #输入层节点
OUTPUT_NODE=10  #输出层节点
LAYER1_NODE=500 #隐藏层节点

#定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 输出层不过激活
    return y