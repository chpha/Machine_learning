import torch
import torch.nn.functional as F

input = torch.tensor([[1.0, 2.0, 3.0],
                      [7.0, 8.0, 9.0],
                      [4.0, 5.0, 6.0],
                      [10.0, 11.0, 12.0]])

kernal = torch.tensor([[1.0, 0.0],
                       [0.0, -1.0]])

input = torch.reshape(input, (1, 1, 4, 3))
kernal = torch.reshape(kernal, (1, 1, 2, 2))

print(input.shape)  # torch.Size([1, 1, 4, 3])
print(kernal.shape)  # torch.Size([1, 1, 2, 2])

output = F.conv2d(input, kernal, stride=1)
print(output)

output2 = F.conv2d(input, kernal, stride=1, padding=1)
print(output2)

output3 = F.conv2d(input, kernal, stride=2, padding=1)
print(output3)  