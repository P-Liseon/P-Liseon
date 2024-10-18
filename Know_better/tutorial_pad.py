'''
torch.pad 是倒着padding,注意是对哪一维padding
'''
import torch
import torch.nn.functional as F

t4d = torch.empty(3, 3, 4, 2)
p1d = (1, 1) # pad last dim by 1 on each side 只对最后一维padding,即2
out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
print(out.size()) # [3, 3, 4, 4]

p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2) (1,1)对倒数第一维，(2,2)对倒数第二维
out = F.pad(t4d, p2d, "constant", 0)
print(out.size()) # [3, 3, 8, 4]

t4d = torch.empty(3, 3, 4, 2)
p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
out = F.pad(t4d, p3d, "constant", 0)
print(out.size())