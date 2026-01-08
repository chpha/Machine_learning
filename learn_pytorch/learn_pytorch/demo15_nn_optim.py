
import torch
import torchvision


dataset = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', train=False, transform = torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = torch.nn.MaxPool2d(2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        return x
    
mynet = Net()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mynet.parameters(), lr=0.001)

for epoch in range(20):
    for data in dataloader:
        imgs, targets = data
        outputs = mynet(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, loss.item()))  

'''
# 实例化神经网络模型
mynet = Net()

# 定义损失函数，这里使用的是交叉熵损失（常用于分类问题）
loss_fn = torch.nn.CrossEntropyLoss()

# 定义优化器，这里使用的是随机梯度下降（SGD），学习率为0.001
optimizer = torch.optim.SGD(mynet.parameters(), lr=0.001)

# 训练过程，循环20个epoch（即遍历数据集20次）
for epoch in range(20):
    # 遍历每一个batch的数据
    for data in dataloader:
        # 获取输入图像和对应标签
        imgs, targets = data

        # 前向传播：将输入数据传入模型，得到输出结果
        outputs = mynet(imgs)

        # 计算损失：比较模型输出和真实标签，得到损失值
        loss = loss_fn(outputs, targets)

        # 梯度清零：每次反向传播前都要清除上一次的梯度
        optimizer.zero_grad()

        # 反向传播：根据损失值计算模型参数的梯度
        loss.backward()

        # 优化器更新参数：根据计算得到的梯度，调整模型参数
        optimizer.step()

    # 每个epoch结束后，打印当前损失值
    print('epoch %d, loss: %f' % (epoch, loss.item()))
    '''   

# 前向传播-->计算损失-->梯度清零-->反向传播-->优化器更新参数