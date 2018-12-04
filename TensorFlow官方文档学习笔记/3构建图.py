import tensorflow as tf
#创建一个常量op，产生一个1x2的矩阵，这个op被作为一个节点
#加到默认图中

#构造器的返回值代表该常量op的返回值
matrix1=tf.constant([[3.,3.]])
# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

#创建一个矩阵乘法matmul op, 把'matrix1','matrix2'作为输入
#返回值'product'代表矩阵乘法的结果
product=tf.matmul(matrix1,matrix2)

# 启动默认图.
sess = tf.Session()
result=sess.run(product)
print(result)

# 任务完成, 关闭会话.
sess.close()