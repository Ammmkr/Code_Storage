import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    """加载MNIST并按8:2划分训练集/验证集（无测试集，匹配用户要求）"""
    # 加载原始数据集（70000样本）
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist.data  # (70000, 784)
    y = mnist.target.astype(np.int64)  # (70000,)

    # 1. 像素值归一化到[0,1]（报告4.1要求）
    X = X / 255.0

    # 2. 8:2划分训练集（56000）/验证集（14000）（无测试集）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("数据集划分完成（无测试集）：")
    print(f"训练集：{X_train.shape[0]} 样本 | 验证集：{X_val.shape[0]} 样本")
    return X_train, X_val, y_train, y_val


def one_hot_encode(y, num_classes=10):
    """纯NumPy实现One-hot编码（报告4.1要求）"""
    m = y.shape[0]
    y_one_hot = np.zeros((m, num_classes))
    y_one_hot[np.arange(m), y] = 1
    return y_one_hot


def create_mini_batches(X, y, batch_size=64, random_state=42):
    """Mini-batch划分（不丢弃最后一个批次，报告4.3要求）"""
    np.random.seed(random_state)
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]
    mini_batches = []

    num_batches = m // batch_size
    for i in range(num_batches):
        X_batch = X_shuffled[i * batch_size: (i + 1) * batch_size]
        y_batch = y_shuffled[i * batch_size: (i + 1) * batch_size]
        mini_batches.append((X_batch, y_batch))

    # 处理最后一个不足批次
    if m % batch_size != 0:
        X_batch = X_shuffled[num_batches * batch_size:]
        y_batch = y_shuffled[num_batches * batch_size:]
        mini_batches.append((X_batch, y_batch))

    return mini_batches


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """计算10×10混淆矩阵（报告5.3.2要求）"""
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_mat[true, pred] += 1
    return conf_mat


def get_error_cases(X, y_true, y_pred, num_cases=10):
    """提取分类错误案例（报告5.3.3要求）"""
    error_indices = np.where(y_true != y_pred)[0]
    np.random.seed(42)
    selected_indices = np.random.choice(error_indices, size=min(num_cases, len(error_indices)), replace=False)
    X_error = X[selected_indices].reshape(-1, 28, 28)  # 还原为28×28图像
    y_true_error = y_true[selected_indices]
    y_pred_error = y_pred[selected_indices]
    return X_error, y_true_error, y_pred_error