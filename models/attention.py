"""Cross-Attention Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class QUEST(nn.Module):
    def __init__(self, attn_dropout=0.1):

        super(QUEST, self).__init__()
        self.in_channel = self.out_channel = 128
        
        self.maxp1 = nn.MaxPool1d(32, stride=32)

        self.temperature = self.out_channel ** 0.5
        self.layer_norm = nn.LayerNorm(self.in_channel)
        proj_dim = 72
        self.q_map = nn.Conv1d(64, proj_dim, 1, bias=False)
        self.k_map = nn.Conv1d(64, proj_dim, 1, bias=False)

        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.fc = nn.Conv1d(self.in_channel, self.out_channel, 1, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, query, supports, prototype, support_x, query_x):
        
        nway, kshot, PN, dim = supports.shape
        batch = query.shape[0]
        way = nway + 1
        residual = prototype
        
        supports = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0).reshape(-1, PN, dim)
        query = self.maxp1(query.transpose(1, 2)).transpose(1, 2)
        supports = self.maxp1(supports.transpose(1, 2)).transpose(1, 2)
        supports = supports.reshape(way, kshot, -1, dim)
        
        proto = 0
        for i in range(kshot):    # N x 5 x PN x C   =>   N x 1 x PN x C
            
            support = supports[:, i, :, :]    # N x PN x C
            
            q = self.q_map(query)   # N x PN x C
            k = self.k_map(support)   #  (N+1) x PN x C
            q = q.reshape(q.shape[1], q.shape[0] * q.shape[2])
            k = k.reshape(k.shape[1], k.shape[0] * k.shape[2])
            
            attn = torch.matmul(q.transpose(0, 1) / self.temperature, k)
            attn = attn.reshape(batch, way, dim, dim)
            attn = F.softmax(attn, dim=-1)

            v = self.v_map(prototype)
            v = v.unsqueeze(2)
            output = torch.matmul(attn, v.transpose(-2, -1)).squeeze(-1).transpose(1, 2)
            output = self.dropout(self.fc(output)).transpose(1, 2)
            output = self.layer_norm(output + residual)
            proto = proto + output / kshot
            
        return output
