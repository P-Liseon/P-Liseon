'''
只需要明白对谁softmax即可
tensor 维度是2
dim = 0 对不同行的同一位置元素softmax,也就是同一列
dim = 1 对不同列的同一位置元素softmax,也就是同一行

一般情况下,不管几维,想对行softmax,dim = -1
'''
import torch
import torch.nn.functional as F
 
data=torch.FloatTensor([[1.0,2.0,3.0],[4.0,6.0,8.0]])
print(data)
print(data.shape)
print(data.type())
 
prob = F.softmax(data,dim=0) # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
print(prob)
print(prob.shape)
print(prob.type())