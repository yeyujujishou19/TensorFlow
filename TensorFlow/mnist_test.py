#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS=5    #程序循环的间隔时间  5s

def test(mnist):   #读入mnist数据集
    with tf.Graph().as_default() as g:  #复现计算图
        #初始化
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        #前向传播计算y的值
        y = mnist_forward.forward(x, None)

        #实例化带滑动平均的saver对象，这样所有参数被加载时，都会被赋值为各自的滑动平均值
        ema=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        #计算准确率
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:
                #加载ckpt，即将滑动平均值赋值给各个参数
                ckpt=tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                #判断有没有模型，如果有，先恢复模型到当前会话
                if ckpt and ckpt.model_checkpoint_path:
                    #先恢复模型到当前会话
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #恢复global_step值
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split("-")[-1]
                    #执行准确率计算
                    accuracy_score=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
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