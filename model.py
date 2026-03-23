import numpy as np


# 激活函数及导数（实验要求：支持ReLU/Sigmoid/Tanh）
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(np.float64)


def sigmoid(x):
    # 防止溢出
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


# Softmax函数（输出层，实验要求：10维Softmax对应0-9分类）
def softmax(x):
    # 防止溢出：每行减去最大值
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 交叉熵损失函数+L2正则项（实验要求：实现交叉熵损失、L2正则）
def cross_entropy_loss(y_pred, y_true, layers, l2_lambda):
    n_samples = y_pred.shape[0]
    # 交叉熵损失（防止log(0)）
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    ce_loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    # L2正则项（所有全连接层的权重W，不惩罚偏置b，实验常规要求）
    l2_loss = 0
    for layer in layers:
        l2_loss += np.sum(layer.W ** 2)
    l2_loss = 0.5 * l2_lambda * l2_loss / n_samples
    return ce_loss + l2_loss


# 全连接层类（实验核心要求：手动实现前向/反向传播，类名/方法名贴合实验步骤）
class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        # 权重初始化（实验默认：randn*0.01；预留Xavier/He初始化接口，进阶优化）
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))  # 偏置初始化为0
        self.activation = activation
        # 保存前向传播中间值，用于反向传播
        self.X = None
        self.Z = None  # 线性输出：Z = X@W + b
        self.A = None  # 激活输出：A = activation(Z)

        # 激活函数及导数映射
        self.act_fun = {
            'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'none': lambda x: x  # 输出层无激活（后续接Softmax）
        }[activation]

        self.act_deriv = {
            'relu': relu_deriv,
            'sigmoid': sigmoid_deriv,
            'tanh': tanh_deriv,
            'none': lambda x: 1  # 输出层导数为1
        }[activation]

    # 前向传播（实验要求：实现全连接层前向）
    def forward(self, X):
        self.X = X  # 保存输入，用于反向传播计算dW
        self.Z = np.dot(X, self.W) + self.b  # 线性变换
        self.A = self.act_fun(self.Z)  # 激活变换
        return self.A

    # 反向传播（实验要求：实现全连接层反向，输入dZ：损失对Z的梯度）
    def backward(self, dZ, l2_lambda):
        n_samples = dZ.shape[0]
        # 计算权重梯度dW：包含L2正则梯度（实验要求：L2正则）
        self.dW = np.dot(self.X.T, dZ) / n_samples + l2_lambda * self.W / n_samples
        # 计算偏置梯度db：无L2正则
        self.db = np.sum(dZ, axis=0, keepdims=True) / n_samples
        # 计算损失对输入X的梯度dX，传递给上一层
        dX = np.dot(dZ, self.W.T)
        return dX


# 多层神经网络主类
class MultiLayerNN:
    def __init__(self, layer_dims, activations):
        # layer_dims：层维度列表，如[784,256,128,10]（实验架构示例）
        # activations：激活函数列表，如['relu','relu','none']
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(FullyConnectedLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                activation=activations[i]
            ))

    # 模型前向传播（整体）
    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        # 输出层接Softmax（实验要求）
        y_pred = softmax(A)
        return y_pred

    # 修复后的反向传播逻辑（核心！！）
    def backward(self, y_pred, y_true, l2_lambda):
        n_samples = y_pred.shape[0]

        # 1. 处理输出层：计算损失对输出层激活的梯度dA
        # Softmax+交叉熵的联合导数：dA = y_pred - y_true（简化推导，维度匹配）
        dA = y_pred - y_true

        # 2. 反向遍历所有层（从输出层到输入层）
        for layer in reversed(self.layers):
            # 2.1 计算损失对当前层线性输出Z的梯度dZ = dA * 激活函数导数
            dZ = dA * layer.act_deriv(layer.Z)

            # 2.2 调用当前层的backward，得到传递给上一层的dA（即损失对当前层输入的梯度）
            dA = layer.backward(dZ, l2_lambda)

    # Mini-batch梯度下降更新参数（实验要求：实现Mini-batch梯度下降优化器）
    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    # 预留进阶优化接口：动量Momentum（实验可选进阶优化）
    def init_momentum(self):
        self.v_W = [np.zeros_like(layer.W) for layer in self.layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.layers]

    def update_params_momentum(self, learning_rate, beta=0.9):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = beta * self.v_W[i] + (1 - beta) * layer.dW
            self.v_b[i] = beta * self.v_b[i] + (1 - beta) * layer.db
            layer.W -= learning_rate * self.v_W[i]
            layer.b -= learning_rate * self.v_b[i]