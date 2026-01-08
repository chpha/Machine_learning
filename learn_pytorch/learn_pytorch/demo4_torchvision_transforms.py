import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # 转为张量
])  # 组合多个变换操作
data_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=data_transforms, download=True)
data_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=data_transforms, download=True)

print(data_train[0])

writer = SummaryWriter("logs")
for i in range(10):
    img, label = data_train[i]
    writer.add_image("train", img, i)  # 将图片数据写入logs文件夹
writer.close()


'''
print(data_train.classes)

img, label = data_train[0]
print(img)
print(label)
print(data_train.classes[label])
img.show()  # 显示图片
'''