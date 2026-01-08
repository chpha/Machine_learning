from PIL import Image
import torchvision
from demo20_model import *

image_path = "/data/ssd0/chenxiaolong/code/learn_pytorch/learn_pytorch/test/dog.png"
image =Image.open(image_path)  # 打开图像
print(image)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)), # 调整图像大小为32x32
    torchvision.transforms.ToTensor() # 将图像转换为张量
])

image_tensor = transform(image)  # 应用转换
print(image_tensor.shape)  # 输出张量的形状, torch.Size([3, 32, 32])

model = torch.load("model_39_gpu.pth")  # 加载模型
print(model)
image = image_tensor.view(1, 3, 32, 32)  # 调整图像张量的形状为 [1, 3, 32, 32]
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不需要计算梯度
    output = model(image)
print(output)  # 输出模型的预测结果