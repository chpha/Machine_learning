from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open('learn_pytorch/learn_pytorch/data/train/ants_image/5650366_e22b7e1065.jpg')
print(img)  # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333 at 0x7F8C1C3B3D90>

# ToTensor
transform1 = transforms.ToTensor()
img_tensor = transform1(img)
writer.add_image('ToTensor Image', img_tensor)
# Normalize
transform2 = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normalized =transform2(img_tensor)
print(img_tensor[0][0][0])  # tensor value before normalization
print(img_normalized[0][0][0])  # tensor value after normalization

writer.add_image('Normalized Image', img_normalized)
writer.close()

