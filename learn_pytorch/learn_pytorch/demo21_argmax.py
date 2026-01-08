'''
1. argmax 的作用
argmax 用于返回张量中最大值的索引。
在深度学习 "分类" 任务中，通常用它来找出每个样本预测概率最大的类别。

2. argmax(dim) 的含义
dim=0：对每一列找最大值索引（跨行比较）。
dim=1：对每一行找最大值索引（跨列比较）。
举例说明
假设有一个模型输出张量 outputs，形状为 [batch_size, num_classes]，比如：
outputs = torch.tensor([
    [0.1, 0.5, 0.4],  # 第一张图片的三个类别得分
    [0.8, 0.1, 0.1],  # 第二张图片的三个类别得分
    [0.3, 0.2, 0.5]   # 第三张图片的三个类别得分
])

outputs.argmax(1)

对每一行找最大值索引（每张图片预测的类别）：
第一行最大值是0.5，索引1
第二行最大值是0.8，索引0
第三行最大值是0.5，索引2
结果：
tensor([1, 0, 2])
表示三张图片分别预测为类别1、类别0、类别2。

outputs.argmax(0)

对每一列找最大值索引（每个类别在所有图片中的最大值出现在哪一行）：
第一列最大值是0.8，出现在第2行（索引1）
第二列最大值是0.5，出现在第1行（索引0）
第三列最大值是0.5，出现在第3行（索引2）
结果：
tensor([1, 0, 2])
表示类别0最大值在第2张图片，类别1最大值在第1张图片，类别2最大值在第3张图片。

3. 分类任务为什么用 argmax(1)
因为我们要对每张图片预测类别，所以要对每一行找最大值索引，即 argmax(1)。

总结
argmax(1)：每张图片预测的类别（常用于分类任务）。
argmax(0)：每个类别在所有图片中的最大值出现在哪张图片（一般不用于分类）。
'''

import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
print("outputs:", outputs)
print("outputs.argmax(0):", outputs.argmax(0))  # tensor([1, 1])  # 每列最大值索引
print("outputs.argmax(1):", outputs.argmax(1))  # tensor([1, 1])  # 每行最大值索引

predt = outputs.argmax(1)
targets = torch.tensor([0, 1])
print("predt == targets:", predt == targets)  # tensor([False,  True])
print("(predt == targets).sum():", (predt == targets).sum())  # tensor(1)  # 预测正确的数量