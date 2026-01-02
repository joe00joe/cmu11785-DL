import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

sys.path.append('./')
DATA_PATH = "./autograder/hw1_autograder/data"
# -------------------------- 1. 配置超参数与设备 --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.1
num_epochs = 10
input_size = 784  # 28*28展平后
hidden_size = 20
num_classes = 10

# -------------------------- 2. 从本地 .npy 文件加载数据集（核心修改） --------------------------
# 关键：修改为你的本地 MNIST .npy 文件存放目录
def load_data():
    train_x = np.load(os.path.join(DATA_PATH, "train_data.npy"))
    train_y = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    val_x = np.load(os.path.join(DATA_PATH, "val_data.npy"))
    val_y = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

    train_x = train_x / 255
    val_x = val_x / 255

    return train_x, train_y, val_x, val_y
train_x_np, train_y_np, test_x_np, test_y_np = load_data()

train_x_np = train_x_np.astype(np.float32)  # 图像数据：float32
train_y_np = train_y_np.astype(np.int64)    # 标签数据：int64（适配CrossEntropyLoss）
test_x_np = test_x_np.astype(np.float32)
test_y_np = test_y_np.astype(np.int64)

# 转换为 PyTorch 张量
train_x = torch.from_numpy(train_x_np)
train_y = torch.from_numpy(train_y_np)
test_x = torch.from_numpy(test_x_np)
test_y = torch.from_numpy(test_y_np)

# 验证数据形状（确保展平为 784 维，适配模型输入）
print(f"训练图像形状：{train_x.shape}")  # 应输出 (60000, 784)
print(f"训练标签形状：{train_y.shape}")  # 应输出 (60000,)
print(f"测试图像形状：{test_x.shape}")   # 应输出 (10000, 784)
print(f"测试标签形状：{test_y.shape}")   # 应输出 (10000,)

# 构建 TensorDataset（PyTorch 支持的数据集格式）
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

# 构建 DataLoader（批量加载数据，与之前逻辑一致）
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True  # 训练集打乱
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False  # 测试集不打乱
)

# -------------------------- 3. 定义模型（与之前完全一致，严格匹配要求） --------------------------
class MNISTNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MNISTNet, self).__init__()
        # Linear(784,20) -> BatchNorm1d(20) -> ReLU() -> Linear(20,10)
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MNISTNet(input_size, hidden_size, num_classes).to(device)

# -------------------------- 4. 损失函数与优化器（与之前完全一致） --------------------------
criterion = nn.CrossEntropyLoss()  # Softmax交叉熵损失（内置Softmax）
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD 优化器，lr=0.1

# -------------------------- 5. 模型训练（与之前完全一致） --------------------------
print("\n开始训练模型...")
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 数据移至设备（GPU/CPU）
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 打印训练信息
        if (i+1) % 300 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # 打印每轮平均损失
    avg_train_loss = running_loss / total_step
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}')

print("模型训练完成！")

# -------------------------- 6. 模型测试（与之前完全一致） --------------------------
print("\n开始测试模型...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the 10000 test images: {test_accuracy:.2f} %')

# -------------------------- 7. 保存模型（可选） --------------------------
torch.save(model.state_dict(), 'mnist_model_local_data.pth')
print("模型已保存为 mnist_model_local_data.pth")