#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np

TEST_INTERVAL_SECS=5    #程序循环的间隔时间  5s

def test(mnist):   #读入mnist数据集
    with tf.Graph().as_default() as g:  #复现计算图
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE])
        y = mnist_lenet5_forward.forward(x,False, None)

        #实例化带滑动平均的saver对象，这样所有参数被加载时，都会被赋值为各自的滑动平均值
        ema=tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        #计算准确率
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:
                #加载ckpt，即将滑动平均值赋值给各个参数
                ckpt=tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
                #判断有没有模型，如果有，先恢复模型到当前会话
                if ckpt and ckpt.model_checkpoint_path:
                    #先恢复模型到当前会话
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #恢复global_step值
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split("-")[-1]
                    reshaped_x = np.reshape(mnist.test.images, (
                    mnist.test.num_examples,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.NUM_CHANNELS))
                    #执行准确率计算
                    accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_x,y_:mnist.test.labels})
                    print("After %s training step(s), test accuracy=%g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")  #未找到模型
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist=input_data.read_data_sets('./data/',one_hot=True) #读入数据集
    test(mnist)  #执行test函数

if __name__=='__main__':
    main()