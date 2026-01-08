
import torch


input = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1, 2, 5], dtype = torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

# MAE
loss_fn1 = torch.nn.L1Loss()
loss1 = loss_fn1(input, target)
print("MAE loss = ", loss1)

# MSE
loss_fn2 = torch.nn.MSELoss()
loss2 = loss_fn2(input, target)
print("MSE loss =", loss2)

# CrossEntropy
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))
loss_f3  = torch.nn.CrossEntropyLoss()
loss3 = loss_f3(x, y)
print("CrossEntropy loss =", loss3)