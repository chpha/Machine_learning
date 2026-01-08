import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torchvision

dataset = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', transform=torchvision.transforms.ToTensor(), train = False, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # 引入Sequential简化
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss_cross = torch.nn.CrossEntropyLoss()
mymodel = model()


for data in dataloader:
    imgs, targets = data
    outputs = mymodel(imgs)
    loss = loss_cross(outputs, targets)
    loss.backward()  # 反向传播
    print("loss =", loss.item())
    
