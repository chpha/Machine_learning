# 创建模型
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, 5, padding=2),  # [b, 3, 32, 32] => [b, 32, 32, 32]
            MaxPool2d(2),  # [b, 32, 32, 32] => [b, 32, 16, 16]
            Conv2d(32, 32, 5, 1, 2),  # [b, 32, 16, 16] => [b, 32, 16, 16]
            MaxPool2d(2),  # [b, 32, 16, 16] => [b, 32, 8, 8]
            Conv2d(32, 64, 5, 1, 2),  # [b, 32, 8, 8] => [b, 64, 8, 8]
            MaxPool2d(2),  # [b, 64, 8, 8] => [b, 64, 4, 4]
            Flatten(), # [b, 64, 4, 4] => [b, 64*4*4]
            Linear(64*4*4, 64),  # [b, 64*4*4] => [b, 64]
            Linear(64, 10)  # [b, 64] => [b, 10] 
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == "__main__":
    model = Model()
    input = torch.ones((64, 3, 32, 32))
    output = model(input)
    print("output:", output.shape)  # [64, 10]