import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    load_mnist, one_hot_encode, create_mini_batches,
    compute_confusion_matrix, get_error_cases
)
from model import NeuralNetwork

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_model(model, X_train, y_train, X_val, y_val, y_train_one_hot, y_val_one_hot,
                epochs=30, batch_size=64, learning_rate=0.01, lambd=1e-4, beta=0.9, random_state=42):
    """模型训练（适配Dropout的training参数）"""
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    m = X_train.shape[0]

    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history
    }

    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train_one_hot, batch_size, random_state)
        train_loss = 0

        # 遍历Mini-batch训练（training=True）
        for X_batch, y_batch in mini_batches:
            y_hat = model.forward_propagation(X_batch, training=True)  # 训练阶段启用Dropout
            loss = model.compute_loss(y_hat, y_batch, lambd)
            train_loss += loss * X_batch.shape[0]
            grads = model.backward_propagation(X_batch, y_hat, y_batch, lambd)
            model.update_parameters(grads, learning_rate, beta)

        # 计算本轮指标（验证阶段training=False）
        avg_train_loss = train_loss / m
        # 训练集准确率：用training=False（和验证集一致，避免Dropout干扰）
        train_acc = model.compute_accuracy(X_train, y_train)
        # 验证集前向传播：training=False
        y_val_hat = model.forward_propagation(X_val, training=False)
        val_loss = model.compute_loss(y_val_hat, y_val_one_hot, lambd)
        val_acc = model.compute_accuracy(X_val, y_val)

        # 记录历史数据
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 每5轮打印进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | 训练损失：{avg_train_loss:.4f} | 验证损失：{val_loss:.4f} | "
                  f"训练准确率：{train_acc:.4f} | 验证准确率：{val_acc:.4f}")

    return model, history

def plot_training_curves(history, title):
    """绘制训练/验证曲线（保留）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_loss'], label='训练损失')
    ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(history['train_acc'], label='训练准确率')
    ax2.plot(history['val_acc'], label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_title('准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.savefig(f'{title}_训练曲线.png', dpi=300, bbox_inches='tight')
    #plt.show()


def plot_confusion_matrix(conf_mat, title):
    """绘制混淆矩阵（保留）"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_mat, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('样本数', rotation=-90, va='bottom')

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_yticklabels([str(i) for i in range(10)])
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)

    # 标注每个单元格的样本数
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    plt.savefig(f'{title}_混淆矩阵.png', dpi=300, bbox_inches='tight')
    #plt.show()


def plot_error_cases(X_error, y_true_error, y_pred_error, title):
    """可视化错误案例（保留）"""
    num_cases = len(X_error)
    n_rows = 2
    n_cols = num_cases // 2 if num_cases % 2 == 0 else num_cases // 2 + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(num_cases):
        axes[i].imshow(X_error[i], cmap='gray')
        axes[i].set_title(f'真实：{y_true_error[i]} | 预测：{y_pred_error[i]}')
        axes[i].axis('off')

    # 隐藏多余子图
    for i in range(num_cases, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.savefig(f'{title}_错误案例.png', dpi=300, bbox_inches='tight')
    #plt.show()

def run_experiments():
    """运行三组实验（为每组配置Dropout率）"""
    # 1. 数据加载与预处理
    X_train, X_val, y_train, y_val = load_mnist()
    y_train_one_hot = one_hot_encode(y_train)
    y_val_one_hot = one_hot_encode(y_val)

    # 2. 实验配置（新增Dropout率）
    experiments = [
        {
            'name': 'Baseline组（Dropout=0.2）',
            'layer_dims': [784, 256, 128, 10],
            'activations': ['relu', 'relu', 'none'],
            'dropout_rates': [0.2, 0.2, 0.0],  # 256层0.2，128层0.2，输出层0
            'learning_rate': 0.01,
            'batch_size': 64,
            'lambd': 1e-4,
            'epochs': 30
        },
        {
            'name': '组1（Dropout=0.2）',
            'layer_dims': [784, 512, 256, 10],
            'activations': ['leaky_relu', 'leaky_relu', 'none'],
            'dropout_rates': [0.2, 0.2, 0.0],  # 512层0.2，256层0.2，输出层0
            'learning_rate': 0.005,
            'batch_size': 128,
            'lambd': 1e-3,
            'epochs': 30
        },
        {
            'name': '组2（Dropout=0.1）',
            'layer_dims': [784, 128, 10],
            'activations': ['tanh', 'none'],
            'dropout_rates': [0.1, 0.0],  # 128层0.1，输出层0（tanh层Dropout率稍低）
            'learning_rate': 0.02,
            'batch_size': 32,
            'lambd': 0.0,
            'epochs': 30
        }
    ]

    # 3. 运行每组实验
    final_results = []
    for exp in experiments:
        print(f"\n==================== 开始训练 {exp['name']} ====================")
        # 初始化模型（传入Dropout率）
        model = NeuralNetwork(
            layer_dims=exp['layer_dims'],
            activations=exp['activations'],
            dropout_rates=exp['dropout_rates']
        )
        # 训练模型
        trained_model, history = train_model(
            model, X_train, y_train, X_val, y_val,
            y_train_one_hot, y_val_one_hot,
            epochs=exp['epochs'],
            batch_size=exp['batch_size'],
            learning_rate=exp['learning_rate'],
            lambd=exp['lambd']
        )
        # 绘制训练曲线
        plot_training_curves(history, exp['name'])

        # 计算最终验证准确率
        val_acc = trained_model.compute_accuracy(X_val, y_val)
        print(f"{exp['name']} 最终验证准确率：{val_acc:.4f}")

        # 记录实验结果
        final_results.append({
            '实验组': exp['name'],
            '隐藏层结构': str(exp['layer_dims'][1:-1]),
            '激活函数': '/'.join(exp['activations'][:-1]),
            'Dropout率': str(exp['dropout_rates'][:-1]),  # 仅显示隐藏层
            '学习率': exp['learning_rate'],
            'Batch Size': exp['batch_size'],
            'L2系数': exp['lambd'],
            '最终验证准确率': val_acc
        })

        # Baseline组绘制混淆矩阵和错误案例
        if 'Baseline' in exp['name']:
            y_pred_val, _ = trained_model.predict(X_val)
            conf_mat = compute_confusion_matrix(y_val, y_pred_val)
            plot_confusion_matrix(conf_mat, exp['name'])
            X_error, y_true_error, y_pred_error = get_error_cases(X_val, y_val, y_pred_val)
            plot_error_cases(X_error, y_true_error, y_pred_error, exp['name'])


    # 5. 实验结果汇总
    results_df = pd.DataFrame(final_results)
    print("\n==================== 三组实验结果汇总（含Dropout） ====================")
    print(results_df.to_string(index=False))
    results_df.to_csv('实验结果汇总_Dropout.csv', index=False, encoding='utf-8-sig')

    # 可视化验证集准确率对比
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df['实验组'], results_df['最终验证准确率'], width=0.6,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('最终验证准确率')
    ax.set_title('三组实验验证准确率对比（动量优化+Dropout+8:2数据划分）')
    ax.set_ylim(0.9, 1.0)
    # 标注准确率数值
    for bar, acc in zip(bars, results_df['最终验证准确率']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig('验证准确率对比_Dropout.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    run_experiments()