import torch
'''
torch.cat 不会增加维度，换句话说，要合并的张量是几维，合并后依然是几维，只改变形状
torch.cat(tensors, dim)注意输入不是tensor,tensor 的元组或列表
'''

a = torch.randn(2, 3, 4)
b = torch.randn(1, 3, 4)
c = torch.cat([a, b], 0)
print(a)
print(b)
print(c)
