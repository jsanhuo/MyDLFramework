from DLFramework.Tensor import Tensor
import numpy as np


# 层的定义
class Layer(object):

    def __init__(self):
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
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


# 序列化
class Sequential(Layer):

    def __init__(self, layers=list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)


    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    # 获得所有参数
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


