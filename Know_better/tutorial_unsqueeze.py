import torch
from torch import nn
import torch.nn.functional as F

'''
torch.unsqueeze(input, dim) 向张量的指定位置插入尺寸为1的新维度
input :Tensor
dim :dimension
'''
x = torch.randn(3, 4) # 标准正太分布
print("原始张量形状：",x.shape) # [3, 4]
#两种写法等价，Tensor数据类型自带unsqueeze方法
# y = torch.unsqueeze(x, 0)
y = x.unsqueeze(0)
print("在第0维插入新维度：", y.shape) # [1, 3, 4]
y = torch.unsqueeze(x, 1)
print("插入第1维: ", y.shape) # [3, 1, 4]
