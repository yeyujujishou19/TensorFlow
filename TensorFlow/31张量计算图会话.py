import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a=tf.constant([[1.0,2.0]])
b=tf.constant([[3.0],[4.0]])

result=tf.matmul(a,b)
print(result)

with tf.Session() as sess:
     print (sess.run(result))