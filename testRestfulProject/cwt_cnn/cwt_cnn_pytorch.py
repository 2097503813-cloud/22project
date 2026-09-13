# coding:utf-8
"""cwt_cnn —— 算法模型2（PyTorch 实现）。

原文件是一个「import 即开始训练」的脚本：只要被 import 就立刻跑 50 个 epoch 并弹图，
因此 model_service 没法复用它。这里做了一次**行为保持不变**的重构：
把训练主体收进函数、CLI 入口放进 `if __name__ == "__main__":`，
`python cwt_cnn_pytorch.py` 的原有表现（50 epoch、打印 loss、分类报告、混淆矩阵）完全不变。

对外暴露：build_model / load_data / train_model / predict / evaluate / main

注意：目录名叫 cwt_cnn，但本文件与原文件一样，**没有任何连续小波变换**——它是普通的 1D-CNN。
改成名副其实（加 pywt 做 CWT 时频图 + 2D-CNN）是另一件事，这里不动，以免悄悄改变既有实验结果。
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

import preprocess  # 需要提供相应的预处理模块

# ---------------------------------------------------------------- 默认超参
num_classes = 10    # 样本类别
length = 784        # 样本长度
number = 300        # 每类样本的数量
normal = True       # 是否标准化
rate = [0.5, 0.25, 0.25]   # 测试集验证集划分比例


class MyModel(nn.Module):
    """结构与原文件一致；仅把 fc1 的输入维度改为按 length 推导（原来硬编码 16*196）。"""

    def __init__(self, num_classes: int = 10, length: int = 784):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * (length // 4), 32)   # 两次 MaxPool1d(2) 后长度为 length//4
        self.fc2 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(num_classes: int = 10, length: int = 784) -> MyModel:
    """建模型（默认值与模块级常量一致；服务侧会显式传参）。"""
    return MyModel(num_classes=int(num_classes), length=int(length))


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: np.ndarray, length_: int, dtype=torch.float32, device=None):
    """(n, length) → (n, 1, length) 张量。"""
    t = torch.tensor(np.asarray(x).reshape(-1, 1, length_), dtype=dtype)
    return t.to(device) if device is not None else t


def load_data(d_path: str = r'0HP', length_: int = length, number_: int = number,
              normal_: bool = normal, rate_: list = None) -> dict:
    """走原有的 preprocess.prepro，口径与命令行脚本完全一致。"""
    rate_ = rate_ or rate
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(
        d_path=d_path, length=length_, number=number_, normal=normal_, rate=rate_,
        enc=False, enc_step=28)
    return {
        "X_train": np.array(x_train), "y_train": np.array(y_train),
        "X_valid": np.array(x_valid), "y_valid": np.array(y_valid),
        "X_test": np.array(x_test), "y_test": np.array(y_test),
    }


def shuffle_xy(x, y, seed: int = 1):
    index = list(range(len(x)))
    random.seed(seed)
    random.shuffle(index)
    return np.array(x)[index], np.array(y)[index]


def train_model(model: MyModel, x_train, y_train, epochs: int = 50, lr: float = 1e-3,
                device=None, verbose_every: int = 10, log=print) -> list:
    """整批（full-batch）梯度下降，与原脚本同口径，返回每个 epoch 的 loss。"""
    device = device or default_device()
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if log and verbose_every and (epoch + 1) % verbose_every == 0:
            log(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return losses


def predict(model: MyModel, x, device=None) -> np.ndarray:
    """返回预测类别（numpy int 数组）。"""
    device = device or default_device()
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(device) if hasattr(x, "to") else x)
        return torch.max(outputs.data, 1)[1].cpu().numpy()


def evaluate(model: MyModel, x_test, y_test, device=None, digits: int = 4):
    """返回 (准确率, classification_report 文本)。"""
    device = device or default_device()
    y_true = y_test.cpu().numpy() if hasattr(y_test, "cpu") else np.asarray(y_test)
    y_pred = predict(model, x_test, device=device)
    accuracy = float((y_pred == y_true).sum()) / float(len(y_true)) if len(y_true) else 0.0
    report = classification_report(y_true, y_pred, digits=digits, zero_division=0)
    return accuracy, report


def main():
    """原命令行行为：50 epoch 训练 + 评估 + 混淆矩阵。"""
    device = default_device()
    data = load_data(d_path=r'0HP', length_=length, number_=number, normal_=normal, rate_=rate)
    x_train, y_train = shuffle_xy(data["X_train"], data["y_train"])
    x_test, y_test = shuffle_xy(data["X_test"], data["y_test"])

    x_train = to_tensor(x_train, length, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_test = to_tensor(x_test, length, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    model = build_model(num_classes=num_classes, length=length).to(device)
    train_model(model, x_train, y_train, epochs=50, device=device)

    accuracy, report = evaluate(model, x_test, y_test, device=device)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(report)

    y_pred = predict(model, x_test, device=device)
    con_mat = confusion_matrix(y_test.cpu(), y_pred)
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
