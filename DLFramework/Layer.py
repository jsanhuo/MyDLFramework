from DLFramework.Tensor import Tensor
import numpy as np


# 层的定义
class Layer(object):

    def __init__(self):
        # 参数集合
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


# 全连接层
class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        # 初始化权重 n_inputs * n_outputs
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / (n_inputs))
        self.weight = Tensor(W, autograd=True)
        # 偏置
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        # 将参数添加到parameters内
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        """
        正向传播
        :param input:
        :return:
        """
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


# 序列化
class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        """
        添加网络层
        :param layer: 待添加的层
        :return:
        """
        self.layers.append(layer)


    def forward(self, input):
        """
        正向传播
        :param input: 网络输入
        :return: 网络输出
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input


    def get_parameters(self):
        """
        获得所有参数
        :return: 返回所有参数的列表
        """
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


# 损失函数层
class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # 通过公式来计算均方误差
        return ((pred - target) * (pred - target)).sum(0)


