
"""
.cuda()适用范围
1. 模型  model = model.cuda()
2. 损失函数  loss_fn = loss_fn.cuda()
3. 数据输入和标签  imgs = imgs.cuda()  targets = targets.cuda()

"""

import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
# from demo20_model import *
from torch.utils.tensorboard import SummaryWriter

# 创建数据集
dataset_train = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', train=True, download=True, transform=torchvision.transforms.ToTensor())
dataset_test = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', train=False, download=True, transform=torchvision.transforms.ToTensor())

print(len(dataset_train))  # 50000
print(len(dataset_test))  # 10000

#利用dataloader加载数据集
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 64)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 64)

# 创建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, 5, padding=2),  # [b, 3, 32, 32] => [b, 32, 32, 32]
            MaxPool2d(2),  # [b, 32, 32, 32] => [b, 32, 16, 16]
            Conv2d(32, 32, 5, 1, 2),  # [b, 32, 16, 16] => [b, 32, 16, 16]
            MaxPool2d(2),  # [b, 32, 16, 16] => [b, 32, 8, 8]
            Conv2d(32, 64, 5, 1, 2),  # [b, 32, 8, 8] => [b, 64, 8, 8]
            MaxPool2d(2),  # [b, 64, 8, 8] => [b, 64, 4, 4]
            Flatten(), # [b, 64, 4, 4] => [b, 64*4*4]
            Linear(64*4*4, 64),  # [b, 64*4*4] => [b, 64]
            Linear(64, 10)  # [b, 64] => [b, 10] 
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
model = Model()
model = model.cuda()  # 将模型加载到GPU中

# 创建损失函数
loss_fn = CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()  # 将损失函数加载到GPU中
# 创建优化器
lr = 1e-2
# 1e-2 = 1 × 10^-2 = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr)

# 设置训练网络一些参数
# 训练轮数
epoch = 20
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0

# tensorboard
writer = SummaryWriter("../train_logs")

# 开始训练
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # 训练步骤开始
    model.train()  # 设置模型为训练模式，启用 BatchNorm 和 Dropout
    for data in dataloader_train:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()  # 将数据加载到GPU中
            targets = targets.cuda()  # 将标签加载到GPU中
        outputs = (model(imgs))
        loss = loss_fn(outputs, targets)  # 计算损失
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_train_step += 1
        if total_train_step % 100 ==0:
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 训练步骤结束

    # 测试步骤开始
    model.eval()  # 设置模型为评估模式，关闭 BatchNorm 和 Dropout
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 不需要计算梯度
        for data in dataloader_test:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()  # 将数据加载到GPU中
                targets = targets.cuda()  # 将标签加载到GPU中
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 计算准确率
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)表示按行取最大值的索引
            total_accuracy += accuracy.item()
    
        print("整体测试集上的loss:{}".format(total_test_loss))
        print("整体测试集上的正确率:{}".format(total_accuracy / 10000)) # 10000是测试集的总数量
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / 10000, total_test_step)
        total_test_step += 1
        torch.save(model.state_dict(), "model_{}.pth".format(i))  # 每一轮都保存模型
        print("模型已保存")
writer.close()






   



