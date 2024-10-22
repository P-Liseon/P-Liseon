import torch
from torch import nn
import torch.nn.functional as F

class WordEmbedding(nn.Embedding):
    def __init__(self, vocab_size, model_dim):
        super(WordEmbedding, self).__init__(vocab_size, model_dim, padding_idx=1)

class PositionEmbedding(nn.Module):
    def __init__(self, model_dim, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        pos_mat = torch.arange(max_len).unsqueeze(1)
        i_mat = torch.arange(0, model_dim, 2)
        self.encoding[:, 0::2] = torch.sin(pos_mat / torch.pow(10000, i_mat/model_dim))
        self.encoding[:, 1::2] = torch.cos(pos_mat / torch.pow(10000, i_mat/model_dim))
    def forward(self, x):
        return self.encoding[:x.size(0), :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tokenembedding = WordEmbedding(vocab_size=vocab_size, model_dim = model_dim)
        self.posEmbedding = PositionEmbedding(model_dim, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    def forward(self,  x):
        pos_emb = self.posEmbedding(x)
        tok_emb = self.tokenembedding(x)
        return self.drop_out(pos_emb + tok_emb)
    
    