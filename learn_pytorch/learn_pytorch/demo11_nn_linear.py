import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(196608, 10)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs, targets = data
    output = torch.reshape(imgs,(1,1,1,-1))
    output = model(output)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1
writer.close()

    