from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
"""
通过transforms解决两个问题：
1、transforms在python中如何使用
2、为什么需要Tensor数据类型
"""

# 1、transforms在python中如何使用
img_path = "learn_pytorch/learn_pytorch/data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")

writer.add_image("Tensor_img", tensor_img)
writer.close()
# 2、为什么需要Tensor数据类型
