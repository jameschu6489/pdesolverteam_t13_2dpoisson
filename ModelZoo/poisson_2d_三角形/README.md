## 问题描述

二维Poisson方程
$$\Delta u =-\sin (4 \pi x) \sin (4 \pi y),~ (x,y) \in \Omega, \qquad u = \frac{1}{3(4\pi)^2}\sin (4 \pi x) \sin (4 \pi y),  ~ (x,y) \in \partial\Omega$$
其中 $\Omega$ 代表求解区域，我们求解了几种常用的几何边界：矩形、圆形、三角形、五边形。


## 运行环境要求

计算硬件：Ascend 计算芯片

计算框架：Mindspore 1.7.0，numpy 1.21.2，matplotlib 3.5.1，scipy 1.5.4



## 代码框架

```
.
└─PINNforPoisson
  ├─README.md
  ├─src
    ├──config.py                      # parameter configuration
    ├──dataset.py                     # dataset
    ├──model.py                       # network structure
    ├──eager_lbfgs.py                 # L-BFGS algorithm
  ├──solve.py                         # train and test
```





## 模型训练

可以直接使用solve.py文件进行PINNs模型训练和求解Poisson方程。在训练过程中，模型的参数和训练过程也会被自动保存

```
python solve.py
```



## MindScience官网

可以访问官网以获取更多信息：https://gitee.com/mindspore/mindscience
