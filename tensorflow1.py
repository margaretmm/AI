import tensorflow as tf
import numpy as np

# 生成样本数据x_data和预期向量y_data
# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
import time

x_data = np.float32(np.random.rand(2, 100)) # 随机输入一个2行100列的二维数组
print("x_data:", x_data)
y_data = np.dot([0.100, 0.200], x_data) + 0.300 #矩阵[0.100, 0.200]与x_data相乘+0.3   生成一个一维数组
#print("y_data:", y_data)

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))#W是一个1行2列的矩阵, 值在[-1,1]之间随机生成
y = tf.matmul(W, x_data) + b# 矩阵W * 矩阵x_data+b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)#学习步长是0.5
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

#测试打印
print("~~~~~~~~~~~~loss~~~~~~~~~~~~~~~")
sess.run(tf.Print(loss,[loss]))
#sess.run(tf.Print( init,[init]))
print("-----------W------")
sess.run(tf.Print(W, [W]))


# 拟合平面
# 被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简单说，epochs指的就是训练过程中数据将被“轮”多少次，就这样。
epochs=40
for step in range(0, epochs):
    sess.run(train)
    if step % 20 == 0:
        print ("step:",step, "  W:",sess.run(W), "  b:", sess.run(b))
        print("~~~~~~~~~~~~loss~~~~~~~~~~~~~~~")
        sess.run(tf.Print(loss, [loss]))
        time.sleep(1)
# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
