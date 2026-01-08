'''
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[-1,2],[-3,4]], dtype=torch.float32)
print(input.shape)
input = torch.reshape(input, (1,1,2,2)) # (N,C,H,W)需要四维张量
print(input.shape)



dataset = torchvision.datasets.CIFAR10(root = '/data/ssd0/chenxiaolong/code/data', train = False, transform = torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

class NLA(torch.nn.Module):
    def __init__(self):
        super(NLA, self).__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, input):

        output = self.sigmoid(output)
        return output

mymodel = NLA()



writer = SummaryWriter("logs")
atep = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs, atep)
    output = mymodel(imgs)
    writer.add_images("output",output, atep)
    atep += 1
writer.close()

'''

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# CIFAR-10 数据集
dataset = torchvision.datasets.CIFAR10(
    root='/data/ssd0/chenxiaolong/code/data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

# 只用 Sigmoid 的激活类
class MySigmoid(torch.nn.Module):
    def __init__(self):
        super(MySigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(input)

mymodel = MySigmoid()

# TensorBoard
writer = SummaryWriter("logs")
step = 0
for imgs, targets in dataloader:
    output = mymodel(imgs)            # Sigmoid 输出 [0,1]
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()
print("✅ TensorBoard 写入完成")
