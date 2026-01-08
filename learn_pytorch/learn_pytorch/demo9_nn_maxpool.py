import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root = '/data/ssd0/chenxiaolong/code/data', train = False, transform = torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self,x):
        x = self.maxpool(x)
        return x
    
mymodel = model()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = mymodel(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1
writer.close()
