import torch
import torchvision
from demo17_module_save import * # 解决办法2


# 加载方式1，加载整个神经网络模型
vgg16_1 = torch.load("vgg16.pth1")
print("vgg16_1:", vgg16_1)

# 加载方式2， 加载神经网络模型参数
vgg16_2 = torch.load("vgg16.pth2")
print("vgg16_2:", vgg16_2)  # 以字典形式打印模型参数
vgg16 = torchvision.models.vgg16(pretrained=False)  # 实例化一个神经网络模型
vgg16.load_state_dict(vgg16_2)  # 将加载的参数放入模型中
print("vgg16:", vgg16)  # 以模型形式打印模型参数

'''
# 陷阱：不能只保存模型参数，然后直接加载使用      
mynet = torch.load("mynet.pth")
print("mynet:", mynet)  # 报错

# 解决方法1：必须定义模型结构，然后再加载参数
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)

    def forward(self, x):
        x = self.conv1(x)
        return x
mynet = Net()

model = torch.load("mynet.pth")
print("mynet:", model)
'''

# 解决方法2 From demo17_module_save.py import *  
model = torch.load("mynet.pth")
print("mynet:", model)