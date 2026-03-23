import numpy as np
import matplotlib.pyplot as plt
import os  # 新增：用于路径和目录创建
from utils import load_mnist, generate_minibatch, calculate_accuracy, compute_confusion_matrix, plot_confusion_matrix, \
    plot_misclassified_samples
from model import MultiLayerNN, cross_entropy_loss

# ========== 全局配置：保存目录 + 分辨率 ==========
SAVE_DIR = "./mnist_experiment_plots"  # 图像保存目录
DPI = 300  # 高清分辨率
# 自动创建保存目录（不存在则创建）
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ========== 三组实验配置（新增save_prefix用于命名） ==========
experiment_configs = [
    {
        "name": "Baseline",
        "layer_dims": [784, 256, 128, 10],
        "activations": ['relu', 'relu', 'none'],
        "lr": 0.01,
        "batch_size": 64,
        "l2_lambda": 1e-4,
        "epochs": 40,
        "color": "blue",
        "linestyle": "-",
        "save_prefix": "Baseline"  # 保存文件前缀
    },
    {
        "name": "Group1",
        "layer_dims": [784, 512, 256, 10],
        "activations": ['relu', 'relu', 'none'],
        "lr": 0.005,
        "batch_size": 128,
        "l2_lambda": 1e-3,
        "epochs": 40,
        "color": "red",
        "linestyle": "--",
        "save_prefix": "Group1"
    },
    {
        "name": "Group2",
        "layer_dims": [784, 128, 10],
        "activations": ['tanh', 'none'],
        "lr": 0.02,
        "batch_size": 32,
        "l2_lambda": 0,
        "epochs": 40,
        "color": "green",
        "linestyle": ":",
        "save_prefix": "Group2"
    }
]


# ========== 训练函数：新增混淆矩阵/错误样本保存 ==========
def train_single_experiment(config, X_train, X_val, y_train, y_val, y_train_raw, y_val_raw):
    model = MultiLayerNN(config["layer_dims"], config["activations"])
    model.init_momentum()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(config["epochs"]):
        train_loss = 0
        train_acc = 0
        batch_num = 0

        for X_batch, y_batch, y_batch_raw in generate_minibatch(
                X_train, y_train, y_train_raw, config["batch_size"]
        ):
            y_pred_batch = model.forward(X_batch)
            loss_batch = cross_entropy_loss(y_pred_batch, y_batch, model.layers, config["l2_lambda"])
            model.backward(y_pred_batch, y_batch, config["l2_lambda"])
            model.update_params_momentum(config["lr"])

            train_loss += loss_batch
            train_acc += calculate_accuracy(y_pred_batch, y_batch_raw)
            batch_num += 1

        train_loss_avg = train_loss / batch_num
        train_acc_avg = train_acc / batch_num

        # 验证集评估
        y_pred_val = model.forward(X_val)
        val_loss_avg = cross_entropy_loss(y_pred_val, y_val, model.layers, config["l2_lambda"])
        val_acc_avg = calculate_accuracy(y_pred_val, y_val_raw)

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc_avg)
        val_accs.append(val_acc_avg)

        if (epoch + 1) % 5 == 0:
            print(
                f"【{config['name']}】Epoch {epoch + 1:2d}/{config['epochs']} | 训练损失：{train_loss_avg:.4f} | 验证损失：{val_loss_avg:.4f} | 训练准确率：{train_acc_avg:.4f} | 验证准确率：{val_acc_avg:.4f}"
            )

    # ========== 保存单组实验的混淆矩阵 ==========
    val_conf_mat = compute_confusion_matrix(y_pred_val, y_val_raw)
    conf_mat_save_path = os.path.join(SAVE_DIR, f"{config['save_prefix']}_confusion_matrix.png")
    plot_confusion_matrix(val_conf_mat, save_path=conf_mat_save_path, dpi=DPI)

    # ========== 保存单组实验的错误样本 ==========
    mis_sample_save_path = os.path.join(SAVE_DIR, f"{config['save_prefix']}_misclassified_samples.png")
    plot_misclassified_samples(X_val, y_pred_val, y_val_raw, save_path=mis_sample_save_path, dpi=DPI)

    return train_losses, val_losses, train_accs, val_accs


# ========== 主函数：新增合并曲线保存 ==========
def run_all_experiments():
    # 加载数据
    X_train, X_val, y_train, y_val, y_train_raw, y_val_raw = load_mnist()
    print(f"数据加载完成：训练集{X_train.shape}，验证集{X_val.shape}")

    all_results = []
    # 运行三组实验
    for cfg in experiment_configs:
        print(f"\n========== 开始运行实验：{cfg['name']} ==========")
        train_losses, val_losses, train_accs, val_accs = train_single_experiment(
            cfg, X_train, X_val, y_train, y_val, y_train_raw, y_val_raw
        )
        all_results.append({
            "name": cfg["name"],
            "color": cfg["color"],
            "linestyle": cfg["linestyle"],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "epochs": cfg["epochs"]
        })
        print(f"========== 实验 {cfg['name']} 运行完成 ==========\n")

    # ========== 绘制并保存合并损失曲线 ==========
    plt.figure(figsize=(10, 5))
    for res in all_results:
        plt.plot(
            range(1, res["epochs"] + 1), res["train_losses"],
            color=res["color"], linestyle=res["linestyle"],
            label=f"{res['name']} - Train Loss"
        )
        plt.plot(
            range(1, res["epochs"] + 1), res["val_losses"],
            color=res["color"], linestyle=res["linestyle"],
            linewidth=2, alpha=0.7,
            label=f"{res['name']} - Val Loss"
        )
    plt.title("三组实验 - 训练/验证损失对比", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # 保存损失曲线
    loss_curve_path = os.path.join(SAVE_DIR, "All_Experiments_Loss.png")
    plt.savefig(loss_curve_path, dpi=DPI, bbox_inches='tight')
    print(f"合并损失曲线已保存至：{loss_curve_path}")
    plt.show()
    plt.close()

    # ========== 绘制并保存合并准确率曲线 ==========
    plt.figure(figsize=(10, 5))
    for res in all_results:
        plt.plot(
            range(1, res["epochs"] + 1), res["train_accs"],
            color=res["color"], linestyle=res["linestyle"],
            label=f"{res['name']} - Train Acc"
        )
        plt.plot(
            range(1, res["epochs"] + 1), res["val_accs"],
            color=res["color"], linestyle=res["linestyle"],
            linewidth=2, alpha=0.7,
            label=f"{res['name']} - Val Acc"
        )
    plt.title("三组实验 - 训练/验证准确率对比", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.ylim(0.85, 1.0)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # 保存准确率曲线
    acc_curve_path = os.path.join(SAVE_DIR, "All_Experiments_Accuracy.png")
    plt.savefig(acc_curve_path, dpi=DPI, bbox_inches='tight')
    print(f"合并准确率曲线已保存至：{acc_curve_path}")
    plt.show()
    plt.close()

    # 输出最终准确率
    print("\n========== 所有实验最终验证准确率 ==========")
    for res in all_results:
        final_val_acc = res["val_accs"][-1]
        print(f"{res['name']}: {final_val_acc:.4f}")

    print(f"\n所有图像已保存至：{os.path.abspath(SAVE_DIR)}")


# ========== 运行主函数 ==========
if __name__ == '__main__':
    # 强制设置中文显示（防止遗漏）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    run_all_experiments()