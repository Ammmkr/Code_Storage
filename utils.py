import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载并预处理MNIST数据集
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(np.int64)
    X = X / 255.0

    # 标签One-hot编码
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    # 8:2划分训练集/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    y_train_raw, y_val_raw = train_test_split(y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, y_train_raw, y_val_raw


# 计算模型准确率
def calculate_accuracy(y_pred, y_true_raw):
    y_pred_label = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true_raw, y_pred_label)


# Mini-batch数据生成器
def generate_minibatch(X, y, y_raw, batch_size):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    y_raw_shuffled = y_raw[indices]
    for i in range(0, n_samples, batch_size):
        yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size], y_raw_shuffled[i:i + batch_size]


# 混淆矩阵计算（纯numpy实现）
def compute_confusion_matrix(y_pred, y_true_raw, num_classes=10):
    y_pred_label = np.argmax(y_pred, axis=1)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true_raw, y_pred_label):
        conf_mat[true_label, pred_label] += 1
    return conf_mat


# 混淆矩阵可视化（新增save_path参数，支持保存）
def plot_confusion_matrix(conf_mat, class_names=None, save_path=None, dpi=300):
    """
    绘制混淆矩阵并可选保存
    :param conf_mat: 混淆矩阵
    :param class_names: 类别名称
    :param save_path: 保存路径（None则不保存）
    :param dpi: 保存分辨率
    """
    if class_names is None:
        class_names = [str(i) for i in range(conf_mat.shape[0])]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(conf_mat, cmap=plt.cm.Blues)
    plt.colorbar(im)

    plt.xticks(np.arange(len(class_names)), class_names, fontsize=10)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=10)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('MNIST分类混淆矩阵', fontsize=14, pad=20)

    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j],
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black",
                     fontsize=8)

    plt.tight_layout()

    # 新增：保存图像（先保存再show，避免空白）
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"混淆矩阵已保存至：{save_path}")

    plt.show()
    plt.close()  # 释放内存


# 分类错误样本可视化（新增save_path参数，支持保存）
def plot_misclassified_samples(X, y_pred, y_true_raw, num_samples=10, save_path=None, dpi=300):
    """
    可视化错误样本并可选保存
    :param X: 样本特征
    :param y_pred: 模型输出概率
    :param y_true_raw: 真实标签
    :param num_samples: 显示样本数
    :param save_path: 保存路径（None则不保存）
    :param dpi: 保存分辨率
    """
    y_pred_label = np.argmax(y_pred, axis=1)
    misclassified_idx = np.where(y_pred_label != y_true_raw)[0]

    if len(misclassified_idx) < num_samples:
        num_samples = len(misclassified_idx)
        if num_samples == 0:
            print("无分类错误样本！")
            return

    np.random.seed(42)
    selected_idx = np.random.choice(misclassified_idx, num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, idx in enumerate(selected_idx):
        img = X[idx].reshape(28, 28)
        true_label = y_true_raw[idx]
        pred_label = y_pred_label[idx]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'真实：{true_label}\n预测：{pred_label}', fontsize=9)
        axes[i].axis('off')

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('MNIST分类错误样本示例', fontsize=12)
    plt.tight_layout()

    # 新增：保存图像
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"错误样本可视化已保存至：{save_path}")

    plt.show()
    plt.close()