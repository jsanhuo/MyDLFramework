import numpy as np


# 实现Tensor
class Tensor(object):

    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        """
        初始化
        :param autograd: 是否自动求导
        :param data: 数据
        :param creators: 包含创建当前张量所用到的所有张量
        :param creation_op: 创建当前张量所用到的指令
        :param id: 当前Tensor的id
        """
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            id = np.random.randint(0, 1000000)
        self.id = id
        # 如果当前张量由其他张量生成，那么要给父张量添加孩子
        if creators is not None:
            # 遍历所有父张量
            for c in creators:
                # 如果当前节点是父张量第一次使用，那么将值记为1
                if self.id not in c.children:
                    c.children[self.id] = 1
                # 如果当前是第n次使用，那么就加1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        """
        检查所有的孩子是否都计算了梯度
        :return: 如果计算了那么就返回True
        """
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        """
        反向传播
        :param grad: 梯度
        :param grad_origin: 子张量，也就是梯度来的张量
        :return:
        """
        if self.autograd:
            # 如果有子张量
            if grad_origin is not None:
                # 如果有子张量但是子张量使用次数却为0，那么说明出错误了，否则当前子张量使用次数减去1
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
            # 如果当前张量还未有梯度，那么就直接等于
            if self.grad is None:
                self.grad = grad
            # 如果当前张量已经有梯度了，那么需要追加
            else:
                self.grad += grad
            # 如果此张量的父张量列表不为空，并且它的所有子张量都已经被计算梯度，那么继续向下传播
            # 此张量是由其他张量计算得出的，并且通过此张量计算的张量的梯度都已计算
            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                # add 的导数都为1
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                # neg的导数 -1 * grad
                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__(), self)
                # sub的导数 被减数为1 减数为-1
                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)
                # mul的导数 第一个运算数的导数是第二个数
                #          第二个运算数的导数是第一个数
                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)
                # mm 矩阵的导数
                if self.creation_op == "mm":
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

    # 加法（已反向传播）
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    # 负数（已反向传播）
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    # 减法（已反向传播）
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    # 乘法（已反向传播）
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    # 求和（已反向传播）
    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    # 扩展（已反向传播）
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    # 转置（已反向传播）
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())

    # 矩阵乘法（已反向传播）
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


# Test
if __name__ == '__main__':
    a = Tensor([[1, 2], [3, 4]], autograd=True)
    b = Tensor([[2], [2]], autograd=True)
    c = Tensor([5, 4, 3, 2, 1], autograd=True)
    print(a, b)
    print(a.mm(b))
    # d = a - b
    # e = b + c
    # f = d + e
    # print(a.id, b.id, c.id, d.id, e.id, f.id)
    # d.backward(Tensor(np.array([1, 1, 1, 1, 1])))
    # print(b.grad)
