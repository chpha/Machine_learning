import torch
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self,input):
        output = input + 1
        return output


mymodule=MyModule()
input = torch.tensor([1.0, 2.0, 3.0])
output = mymodule(input)
print(output)  # tensor([2., 3., 4.])