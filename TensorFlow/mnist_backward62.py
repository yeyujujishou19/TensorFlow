#coding:utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward62
import os
import mnist_generateds62   ##----##



BATCH_SIZE=200
LEARNING_RATE_BASE=0.1    #最初学习率
LEARNING_RATE_DECAY=0.99  #学习率衰减率
REGULARIZER=0.0001   #正则化系数
STEPS=50000          #训练多少轮
MOVING_AVERAGE_DECAY=0.99   #滑动平均衰减率
MODEL_SAVE_PATH="./model"   #模型保存路径
MODEL_NAME="mnist_model"    #模型名称
train_num_examples=60000  ##----##

def backward():  ##----##
    x = tf.placeholder(tf.float32, [None,mnist_forward62.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,mnist_forward62.OUTPUT_NODE])
    y=mnist_forward62.forward(x,REGULARIZER)

    # 运行了几轮BATCH_SIZE的计数器，初值给0，设为不被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))

    # 定义指数下降学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples/BATCH_SIZE,     ##----##
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义反向传播方法;不含正则化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()
    img_batch,label_batch=mnist_generateds62.gettfrecord(BATCH_SIZE,isTrain=True) ##----##

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH) ##----##
        if ckpt and ckpt.model_checkpoint_path:       ##----##
            saver.restore(sess,ckpt.model_checkpoint_path)    ##----##

        coord=tf.train.Coordinator()    ##----##
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)   ##----##

        for i in range(STEPS):
            xs,ys=sess.run([img_batch,label_batch]) ##----## #xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print("After %d training step(s),loss on training batch is %g."%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

        coord.request_stop() ##----##
        coord.join(threads) ##----##

def main():
    backward()  ##----##

if __name__=='__main__':
    main()