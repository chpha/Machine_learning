

# 测试集
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

data_test = dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset=data_test, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试集中第一个数据集
img, tar = data_test[0]
print(img.shape)  # torch.Size([3, 32, 32])
print(tar)  # 3

writer = SummaryWriter("dataloader")
step = 0
for data in dataloader:
    imgs, tars = data
    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(tars) 
    writer.add_images("test", imgs, step)  # 将图片数据写入logs文件夹
    step += 1
    
writer.close() 


