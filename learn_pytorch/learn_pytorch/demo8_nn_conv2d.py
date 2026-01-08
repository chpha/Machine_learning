import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root = '/data/ssd0/chenxiaolong/code/data', train = False, transform = torchvision.transforms.ToTensor(),download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6,  kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
    
model = mymodel()
print(model)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    outputs = model(imgs)
    # print(outputs.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)
    outputs = torch.reshape(outputs, (-1, 3, 30, 60))
    writer.add_images("output", outputs, step)

    step += 1