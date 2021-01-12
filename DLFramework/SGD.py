class SGD(object):

    def __init__(self, parameters, alpha=0.1):
        """
        SGD初始化
        :param parameters: 需要学习的参数
        :param alpha: 学习率
        """
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        """
        梯度清0
        :return:
        """
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        """
        梯度更新一步
        :param zero: 如果zero = True 则梯度清零
        :return:
        """
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if (zero):
                p.grad.data *= 0