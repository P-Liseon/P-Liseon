import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


# step 1
#关于word embedding,以序列建模为例
#考虑source sentence 和target sentence
#构建序列，序列的字符以其在词表中的索引表示
batch_size = 2 #表示有两个句子

#单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8 # 特征大小，也就是每个单词由几个维度表示

# src_len = torch.randint(2, 5, (batch_size, ))
# tgt_len = torch.randint(2, 5, (batch_size,))
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)
# tensor([2, 4]) 表示第一句子长度为2，第二个句子长度为4
# tensor([4, 3]) 表示目标句子第一个句子长度为4，第二个句子长度为3

# 单词索引构成的源句子和目标句子，并且做了padding，默认值为0 （所以在单词编码的时候，应该从1开始）
# 在输入中每一个句子的长度可能不一致，所以需要一致长度，对输入的句子进行Padding  (0, max(src_len)-L):左边填充0位，右边填充max(src_len)-L位
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max(src_len)-L)), 0) for L in src_len], 0) #单词索引构成的句子
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max(tgt_len)-L)), 0) for L in tgt_len], 0)


# 构造word embedding
src_embedding_table = nn.Embedding(max_num_src_words, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words, model_dim)
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)


# step 2
#构造position embedding
# 在位置编码中，pos 代表的是行，i 代表的是列
#位置编码简易版
# sin(0 / 0_mat) cos(0 / 1_mat) sin(0 / 2_mat)
# sin(1 / 0_mat) cos(1 / 1_mat) xin(1 / 2_mat)

pos_mat = torch.arange(max(src_len)).reshape(-1, 1) # 每一行一样
i_mat = torch.pow(10000, torch.arange(0, model_dim, 2).reshape(1, -1)/model_dim) # 每一列一样 
pe_embedding_table = torch.zeros(max(src_len), model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# 获取每条序列的位置编码
pe_embedding = nn.Embedding(max(src_len), model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad = False)
src_pos =torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len])
tgt_pos =torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len])
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
# print(src_pe_embedding)
# print(tgt_pe_embedding)


# step 3
# 构造encode中self-attention mask
# mask shape: [batch, max_src_len, max_src_len],值为1或-inf
valid_encoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len], 0), 2)
#每个样本（句子）的邻接矩阵
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -np.inf)
prob = F.softmax(masked_score, -1)
# print(prob)



# step 4
# 构造intra_attention_mask   对任何Padding的部分不做attention
# Q * K^T  shape:[batch_size, tgt_seq_len, src_seq_len]
valid_encoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len], 0), 2)
valid_decoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len) - L)), 0) for L in tgt_len], 0), 2)
# 计算encoder中scr句子和decoder中tgt句子的邻接矩阵,得到的是目标句子的每个单词对源句子的有效性
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(-1, -2) )

# print(valid_encoder_pos)
# print(valid_decoder_pos)
# print(valid_cross_pos_matrix)
'''
源句子中单词数目
tensor([[[1.],
         [1.],
         [0.],
         [0.]],

        [[1.],
         [1.],
         [1.],
         [1.]]])
目标句子中单词数目
tensor([[[1.],
         [1.],
         [1.],
         [1.]],

        [[1.],
         [1.],
         [1.],
         [0.]]])
目标句子中的单词对源句子的有效性
矩阵一第一行表示的是 tgt中第一句话中第一个单词对src第一个句子的有效性
矩阵二第三行表示的是 tgt中第二句话中第三个单词对src第二个句子的有效性
tensor([[[1., 1., 0., 0.],
         [1., 1., 0., 0.],
         [1., 1., 0., 0.],
         [1., 1., 0., 0.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [0., 0., 0., 0.]]])
'''
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)  
# print(mask_cross_attention)



# step 5:构造decoder self_attention的mask
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones(L, L)), (0, max(tgt_len)-L, 0, max(tgt_len)-L)), 0) for L in tgt_len], 0)
print(valid_decoder_tri_matrix) #decoder 第一行表示 第一个输入对decoder序列整体的相关性
invalid_decoder_tri_matrix = 1-valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
print(invalid_decoder_tri_matrix)
score =  torch.randn(batch_size, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(masked_score, -1)
print(prob)

# step 6:构建self_attention
def scaled_dot_product_attention(Q, K, V, attn_mask):
    # shape of Q,K,V :[batch_size*num_head, seq_len, model_dim/num_head]
    score = torch.bmm(Q, K.transpose(-1,-2))/torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context