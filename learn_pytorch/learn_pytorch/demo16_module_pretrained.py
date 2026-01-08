import torch
import torchvision


vgg16_true = torchvision.models.vgg16(pretrained=True)
print("true:", vgg16_true)
vgg16_false = torchvision.models.vgg16(pretrained=False)
print("false:", vgg16_false)

dataset = torchvision.datasets.CIFAR10(root='/data/ssd0/chenxiaolong/code/data', train=True, 
                                       transform = torchvision.transforms.ToTensor(), download=True)
vgg16_true.classifier[6] = torch.nn.Linear(4096, 10)
print("true:", vgg16_true)

vgg16_true.classifier.add_module("flatten", torch.nn.Flatten())
print("true:", vgg16_true)

vgg16_false.classifier[6] = torch.nn.Linear(4096, 10)
print("false:", vgg16_false)