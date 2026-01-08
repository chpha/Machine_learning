from PIL import Image
import torch
import torchvision
from demo20_model import *  # 你的模型定义

# 1. 打开图像并处理
image_path = "/data/ssd0/chenxiaolong/code/learn_pytorch/learn_pytorch/test/airplane2.png"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.Grayscale(num_output_channels=3),  # 保证3通道
    torchvision.transforms.ToTensor()
])

image_tensor = transform(image).unsqueeze(0)  # [1, 3, 32, 32]

# 2. 定义模型结构
model = Model()  # Model() 是 demo20_model.py 中的模型类
# 3. 加载 state_dict 权重
state_dict = torch.load("model_39_gpu.pth", map_location=torch.device('cuda'))
model.load_state_dict(state_dict)

# 4. 设置为评估模式
model.eval()

# 5. 预测
with torch.no_grad():
    output = model(image_tensor)

print(output)
print("预测结果的类别索引:", output.argmax(1).item())  # 输出预测类别索引
