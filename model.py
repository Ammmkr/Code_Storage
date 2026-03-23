import numpy as np


class FullyConnectedLayer:
    """全连接层（修复数值溢出+Dropout无效值）"""

    def __init__(self, input_dim, output_dim, activation='relu', init_type='he', dropout_rate=0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.init_type = init_type
        # Dropout核心参数（增加范围校验）
        self.dropout_rate = np.clip(dropout_rate, 0.0, 0.999)  # 限制dropout率≤0.999，避免keep_prob=0
        self.keep_prob = 1 - self.dropout_rate
        self.dropout_mask = None

        self.W, self.b = self._initialize_params()
        self.X = None
        self.Z = None
        self.A = None

        # Momentum缓存
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def _initialize_params(self):
        """参数初始化（限制初始权重范围，防止溢出）"""
        if self.init_type == 'random':
            W = np.random.randn(self.input_dim, self.output_dim) * 0.01
        elif self.init_type == 'xavier':
            W = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(1 / self.input_dim)
        elif self.init_type == 'he':
            W = np.random.randn(self.input_dim, self.output_dim) * np.sqrt(2 / self.input_dim)
        else:
            raise ValueError(f"不支持的初始化方式：{self.init_type}")

        # 裁剪初始权重，防止初始值过大
        W = np.clip(W, -1.0, 1.0)
        b = np.zeros((1, self.output_dim))
        return W, b

    def _activation(self, Z):
        """激活函数（增加数值裁剪，防止inf）"""
        Z = np.clip(Z, -1e6, 1e6)  # 限制Z的范围，避免激活函数输出inf
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # 进一步限制sigmoid输入
        elif self.activation == 'tanh':
            return np.tanh(np.clip(Z, -500, 500))  # 限制tanh输入
        elif self.activation == 'leaky_relu':
            return np.maximum(0.01 * Z, Z)
        elif self.activation == 'none':
            return Z
        else:
            raise ValueError(f"不支持的激活函数：{self.activation}")

    def _activation_deriv(self, Z):
        """激活函数导数（数值稳定版）"""
        Z = np.clip(Z, -1e6, 1e6)
        if self.activation == 'relu':
            return (Z > 0).astype(np.float32)
        elif self.activation == 'sigmoid':
            A = self._activation(Z)
            return A * (1 - A)
        elif self.activation == 'tanh':
            A = self._activation(Z)
            return 1 - A ** 2
        elif self.activation == 'leaky_relu':
            deriv = np.ones_like(Z)
            deriv[Z < 0] = 0.01
            return deriv
        elif self.activation == 'none':
            return np.ones_like(Z)
        else:
            raise ValueError(f"不支持的激活函数：{self.activation}")

    def forward(self, X, training=True):
        """前向传播（修复Dropout无效值）"""
        self.X = np.clip(X, -1e6, 1e6)  # 输入裁剪
        self.Z = np.dot(self.X, self.W) + self.b
        self.A = self._activation(self.Z)

        # Dropout执行逻辑（增加数值校验）
        if self.dropout_rate > 0 and training and self.activation != 'none':
            # 生成掩码（确保mask无nan/inf）
            self.dropout_mask = np.random.rand(*self.A.shape) < self.keep_prob
            self.dropout_mask = np.nan_to_num(self.dropout_mask)  # 替换nan为0
            # 安全计算：防止除以0 + 裁剪输出
            self.A = np.nan_to_num(self.A) * self.dropout_mask / (self.keep_prob + 1e-8)  # +1e-8防除0
            self.A = np.clip(self.A, -1e6, 1e6)  # 裁剪输出
        elif self.dropout_rate > 0 and not training and self.activation != 'none':
            self.A = self.A * self.keep_prob
            self.A = np.clip(self.A, -1e6, 1e6)

        return self.A

    def backward(self, dZ):
        """反向传播（数值稳定版）"""
        # 裁剪梯度，防止梯度爆炸
        dZ = np.clip(dZ, -1e3, 1e3)
        m = max(self.X.shape[0], 1)  # 防止m=0

        # 应用Dropout掩码（安全计算）
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            dZ = np.nan_to_num(dZ) * self.dropout_mask / (self.keep_prob + 1e-8)
            dZ = np.clip(dZ, -1e3, 1e3)

        # 基础梯度计算（防止数值过大）
        dW = (1 / m) * np.dot(self.X.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        # 裁剪梯度
        dW = np.clip(dW, -1e3, 1e3)
        db = np.clip(db, -1e3, 1e3)
        # 上一层梯度传递
        dZ_prev = np.dot(dZ, self.W.T)
        dZ_prev = np.clip(dZ_prev, -1e3, 1e3)

        return dZ_prev, dW, db


class NeuralNetwork:
    """神经网络模型（增加权重裁剪+稳定L2损失）"""

    def __init__(self, layer_dims, activations, init_type='he', dropout_rates=None):
        self.layer_dims = layer_dims
        self.activations = activations
        self.init_type = init_type
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.2] * (len(layer_dims) - 2) + [0.0]
        self.layers = []
        self.num_layers = len(layer_dims) - 1

        # 构建层
        for i in range(self.num_layers):
            dropout_rate = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.0
            self.layers.append(FullyConnectedLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i + 1],
                activation=activations[i],
                init_type=init_type,
                dropout_rate=dropout_rate
            ))

    def _softmax(self, Z):
        """数值稳定的Softmax（防止溢出）"""
        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)  # 减去最大值防止溢出
        sum_exp = np.sum(exp_Z, axis=1, keepdims=True) + 1e-8  # 防除0
        return exp_Z / sum_exp

    def forward_propagation(self, X, training=True):
        """整体前向传播（数值稳定）"""
        A = np.clip(X, -1e6, 1e6)
        for layer in self.layers:
            A = layer.forward(A, training=training)
        if self.layers[-1].activation == 'none':
            A = self._softmax(self.layers[-1].Z)
        return A

    def compute_loss(self, y_hat, y_true, lambd=0.0):
        """带L2正则化的损失（修复平方溢出）"""
        m = max(y_true.shape[0], 1)
        # 裁剪预测值，防止log(0)或log(inf)
        y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
        # 交叉熵损失
        cross_entropy = - (1 / m) * np.sum(y_true * np.log(y_hat))
        cross_entropy = np.clip(cross_entropy, 0, 1e6)  # 裁剪损失

        # L2正则化（修复平方溢出：先裁剪权重再平方）
        l2_loss = 0
        for layer in self.layers:
            # 先裁剪权重到合理范围，再计算平方和
            W_clipped = np.clip(layer.W, -1e3, 1e3)
            l2_loss += np.sum(np.square(W_clipped))
        l2_loss = (lambd / (2 * m)) * l2_loss
        l2_loss = np.clip(l2_loss, 0, 1e6)  # 裁剪L2损失

        total_loss = cross_entropy + l2_loss
        return total_loss

    def backward_propagation(self, X, y_hat, y_true, lambd=0.0):
        """整体反向传播（数值稳定）"""
        m = max(X.shape[0], 1)
        grads = {'dW': [], 'db': []}
        # 输出层梯度（裁剪）
        dZ = y_hat - y_true
        dZ = np.clip(dZ, -1e3, 1e3)

        # 反向传播
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            if i == self.num_layers - 1:
                dZ_prev, dW, db = layer.backward(dZ)
            else:
                dZ_prev, dW, db = layer.backward(dZ)

            # L2正则化梯度（裁剪）
            dW += (lambd / m) * np.clip(layer.W, -1e3, 1e3)
            dW = np.clip(dW, -1e3, 1e3)
            db = np.clip(db, -1e3, 1e3)

            grads['dW'].insert(0, dW)
            grads['db'].insert(0, db)
            dZ = dZ_prev

        return grads

    def update_parameters(self, grads, learning_rate=0.01, beta=0.9):
        """Momentum更新（增加权重裁剪，防止权重爆炸）"""
        for i in range(self.num_layers):
            layer = self.layers[i]
            # 动量累积（裁剪）
            layer.v_W = beta * layer.v_W + (1 - beta) * grads['dW'][i]
            layer.v_b = beta * layer.v_b + (1 - beta) * grads['db'][i]
            layer.v_W = np.clip(layer.v_W, -1e3, 1e3)
            layer.v_b = np.clip(layer.v_b, -1e3, 1e3)

            # 参数更新 + 裁剪权重，核心防止权重溢出
            layer.W -= learning_rate * layer.v_W
            layer.b -= learning_rate * layer.v_b
            # 关键：裁剪权重到合理范围，彻底解决平方溢出
            layer.W = np.clip(layer.W, -10.0, 10.0)
            layer.b = np.clip(layer.b, -10.0, 10.0)

    def predict(self, X):
        """预测（数值稳定）"""
        y_hat = self.forward_propagation(X, training=False)
        return np.argmax(y_hat, axis=1), y_hat

    def compute_accuracy(self, X, y_true):
        """计算准确率"""
        y_pred, _ = self.predict(X)
        return np.mean(y_pred == y_true)