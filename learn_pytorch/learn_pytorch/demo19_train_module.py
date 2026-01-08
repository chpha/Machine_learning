
import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from demo20_model import *
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
model = Model()

# 创建损失函数
loss_fn = CrossEntropyLoss()

# 创建优化器
lr = 1e-2
# 1e-2 = 1 × 10^-2 = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr)

# 设置训练网络一些参数
# 训练轮数
epoch = 40
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
    torch.save(model.state_dict(), "model_{}_gpu.pth".format(i))  # 每一轮都保存模型
    print("模型已保存")

writer.close()



# 准确率相关笔记
'''
1. outputs.argmax(1) 的含义
outputs 是模型对一个 batch 的所有图片的预测结果，形状一般是 [batch_size, num_classes]，比如 [64, 10]。
每一行是一个图片的10个类别的得分（未归一化概率）。
outputs.argmax(1) 表示对每一行（每张图片），找到得分最大的类别索引（即模型认为最可能的类别）。
举例：如果某一行是 [0.1, 0.2, 0.3, ..., 0.05]，那么 argmax(1) 得到的就是 2（因为0.3最大）。
2. (outputs.argmax(1) == targets)
targets 是每张图片的真实类别标签，形状也是 [batch_size]。
outputs.argmax(1) == targets 会得到一个布尔型的张量，表示每张图片预测是否正确。
举例：[True, False, True, ...]
3. .sum()
对布尔张量求和，True 会被当作1，False 当作0。
得到的是当前 batch 中预测正确的图片数量。
4. total_accuracy += accuracy.item()
把每个 batch 的正确数量累加，最后得到整个测试集预测正确的总数。
5. total_accuracy / 10000
10000是测试集总图片数。
用正确数量除以总数，得到准确率（百分比）。
'''


   



