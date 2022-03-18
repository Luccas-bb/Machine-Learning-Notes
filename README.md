# Machine-Learning-Notes
Yue's Machine Learning Notes

#### 感知机（Perception Machine）

$$
f(x)=sign(\boldsymbol{w}^T\boldsymbol{x}+b)
$$

感知机可以表示与门（AND）、与非门（NAND）、或门（OR），两层感知机可以表示异或门（XOR）。单层感知机只能表示线性空间，多层感知机可以表示非线性空间。多层感知机理论上可以表示计算机。

**误分类点到超平面的距离**
$$
-\frac{1}{||\boldsymbol{w}||}|\boldsymbol{w}^T\boldsymbol{x}+b|
$$
对于误分类的点 $-y_i(\boldsymbol{w}^T\boldsymbol{x_i}+b) > 0$

**损失函数**
$$
L(\boldsymbol{w},b) = - \sum_{x_i\in M}y_i(\boldsymbol{w}^T\boldsymbol{x_i}+b)
$$
**随机梯度下降（Stochastic gradient descent，SGD）**

随机选取一个误分类点，根据该点损失函数的负梯度调整 $\boldsymbol{w},b$ （**算法收敛性**）

简单来说，选取一个误分类点向该点靠近。



#### 神经网络

##### Q1常见的激活函数

激活函数是神经网络中非线性的来源,如果只剩下线性运算，最终效果相当于单层线性模型。

**阶跃函数(0,1)**

**Sigmoid函数**

- 相较于阶跃函数，sigmoid函数的平滑性对神经网络的学习具有重要意义。
- 左端趋近于0，右端趋近于1，且两端都趋于[饱和](https://www.cnblogs.com/tangjicheng/p/9323389.html).
- 梯度消失/梯度爆炸
- 非0均值，导数小于0.25

$$
sigmoid(x)=\frac{1}{1+e^{-z}}
$$

![img](https://pic4.zhimg.com/80/v2-2d93251eb7641494b7268fbd8edd888f_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-995ba930f6f2c5dd1ca4ddeb10661666_1440w.jpg)

**tanh函数**

- 它解决了Sigmoid函数的不是zero-centered输出问题，然而，梯度消失（gradient vanishing）的问题和幂运算的问题仍然存在。

$$
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

![img](https://pic4.zhimg.com/80/v2-db9d7889d408a1a13d49be058c797f33_1440w.jpg)

**ReLU(Rectified Linear Unit）**

- 一半的空间是不饱和的
- 解决了梯度消失问题 
- 计算速度非常快
- 收敛速度远快于sigmoid和tanh
- 深度大于宽度：非线性性很弱，因此网络一般要做得很深。更深的模型泛化能力更好。
- 输出不是zero-contered
- Dead ReLU:初始化和Learning Rate
- Leaky ReLU: $LeakyRelu(x)=max(x,\alpha x)$ 

$$
relu(x)=max(x,0)
$$



![这里写图片描述](https://img-blog.csdn.net/20180503231727530?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3R5aGpfc2Y=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### Q2输出层的激活函数

- 恒等函数 回归
- sigmoid 二分类
- softmax 多分类
