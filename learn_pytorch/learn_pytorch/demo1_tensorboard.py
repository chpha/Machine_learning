from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "learn_pytorch/learn_pytorch/Dataset/train/ants/5650366_e22b7e1065.jpg"
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)
print(type(image_array))
print(image_array.shape)

writer.add_image("test", image_array, 5, dataformats='HWC')

# writer.add_image()
for i in range(100):
    writer.add_scalar("y = x", 3*i, i)

writer.close()



'''
from torch.utils.tensorboard import SummaryWriter

# 创建日志写入器，日志会存放在 runs/experiment1 目录下
writer = SummaryWriter("runs")

# 假设你在训练循环中：
for epoch in range(10):
    train_loss = 0.01 * (10 - epoch)  # 假设的损失
    acc = 0.1 * epoch  # 假设的准确率

    # 写入标量数据
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", acc, epoch)

# 关闭写入器
writer.close()
'''