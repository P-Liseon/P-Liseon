import torch

'''
The given dimensions dim0 and dim1 are swapped.
torch.transpose(input, dim0, dim1)
交换dim0 和 dim1
'''
x = torch.randn(2, 3)
print(x)
x_t = torch.transpose(x, 1, 0)
print(x_t)

y = torch.randn(2, 3, 2)
print(y.shape)
y_t = torch.transpose(y, 1, 2)
print(y_t.shape)