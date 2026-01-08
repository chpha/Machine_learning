
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        '''
        self.conv1 = Conv2d(3, 32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(64*4*4, 64)
        self.linear2 = Linear(64, 10)
        '''
        # 引入Sequential简化
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):

        '''
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        '''

        x = self.model1(x)
        return x
    
mymodel = model()
print(mymodel)
input = torch.ones((64, 3, 32, 32))
output = mymodel(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(mymodel, input)
writer.close()