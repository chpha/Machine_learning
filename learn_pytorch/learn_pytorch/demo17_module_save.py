import torch
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1，保存整个神经网络模型：模型结构+模型参数
torch.save(vgg16, "vgg16.pth1")

# 保存方式2，保存神经网络模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16.pth2")

# 陷阱：不能只保存模型参数，然后直接加载使用

class Net(torch.nn.Module):
    def __init(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)

    def forward(self, x):
        x = self.conv1(x)
        return x

mynet = Net()
torch.save(mynet, "mynet.pth")