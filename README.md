机器学习--概念理解&样例


1. 监督学习算法&无监督学习
1.1 监督学习：需要标签系统指导学习的算法，主要解决2类场景的问题：回归问题&分类问题

1.2 无监督学习：不需要标签系统指导学习，一般解决聚类问题的场景，像GAN（本人使用也有限）

2. 回归问题&分类问题：
2.1 回归问题： 一个函数能根据输入信息，输出我们想要的连续的数值，比如房价预测模型

(房价预测样例  ：https://github.com/margaretmm/pricePrediction)

常见的线性回归类型有：

      单变量线性回归： 形如h(x)=theta0+theta1*x1

      多变量线性回归： 形如h(x)=theta0+theta1*x1+theta2*x2+theta3*x3

      多项式回归（Polynomial Regression）： 形如h(x)=theta0+theta1*x1+theta2*(x2^2)+theta3*(x3^3) 或者h(x)=ttheta0+theta1*x1+theta2*sqr(x2) 

2.2 分类问题： 一个函数能根据输入信息输出一些离散值的输出（枚举值），比如判断一个图片是否是猫的模型函数？或者判断某个房子是否容易卖出？

3. 线性回归&逻辑分类 算法
3.1 线性回归： 解决回归问题的算法， 可以使用梯度下降方法， 也可以使用正规方程方法（用的不多不太了解，暂无样例）

3.1.1 单变量线性回归之---最小二乘法拟合二元多次曲线样例：

https://github.com/margaretmm/AI/blob/master/polyfit/polyfit.py

3.1.2 梯度下降法个人样例：https://github.com/margaretmm/pricePrediction/blob/master/tensorflowTrain.py


3.2 逻辑分类：解决分类问题的算法，一般是（线性函数+激活函数）+ 梯度下降 步骤组成

回归问题NN的个人样例（NN网络中的隐藏层使用的是逻辑分类）：https://github.com/margaretmm/pricePrediction/blob/master/tensorflowTrain.py

4. 常规的机器学习算法&神经网络
4.1 神经网络：常用于特征特别多（>1000），识别精准度要求高的场景，NN的特征提取由多层网络（逻辑回归、线性回归等）组成

--房价预测回归问题NN的个人样例：https://github.com/margaretmm/pricePrediction/blob/master/tensorflowTrain.py

--还有针对图像识别的NN如CNN

样例1--图像识别在线预测：https://github.com/margaretmm/TestPlatformOnAI/blob/master/Tensorflow_prediction.java

样例2--图像识别离线模型训练：

https://github.com/margaretmm/TestPlatformOnAI/blob/master/train.py

--针对时序关系的NN如 RNN（没用过）

4.2 常规的机器学习算法：用于特征个数不多，而且训练数据量不大， 精准度要求不高的场景，常见的机器学习算法：

4.2.1 随机森林：

解决回归问题的样例https://github.com/margaretmm/pricePrediction/blob/master/preditcor_RandomForest.py

4.2.2 决策树：(暂无样例)

4.2.3 基于核的算法--支持向量机（SVM）：(暂无样例)
