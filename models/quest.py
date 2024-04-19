"""QUEST Module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class QUEST(nn.Module):
    def __init__(self):

        super(QUEST, self).__init__()
        self.in_channel = self.out_channel = 128
        
        self.maxpool = nn.MaxPool1d(32, stride=32)

        self.layer_norm = nn.LayerNorm(self.in_channel)
        self.proj_dim = 72
        
        self.map = nn.Conv1d(64, self.proj_dim, 1, bias=False)
        self.proto_map = nn.Linear(self.in_channel, self.out_channel)
        
        self.reweight = nn.Linear(self.in_channel, 1, bias=False)

        self.fc = nn.Linear(self.in_channel, self.out_channel, bias=False)
        
    def forward(self, query, supports, prototype):
        
        nway, kshot, PN, dim = supports.shape
        batch = query.shape[0]
        way = nway + 1
        residual = prototype
        
        supports = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0).reshape(-1, PN, dim)
        query = self.maxpool(query.transpose(1, 2)).transpose(1, 2)
        supports = self.maxpool(supports.transpose(1, 2)).transpose(1, 2)
        supports = supports.reshape(way, kshot, -1, dim)
        
        proto = 0
        for i in range(kshot): 
                        
            support = supports[:, i, :, :]
            
            que = self.map(query)
            sup = self.map(support)
            new_proto = self.proto_map(prototype)
            
            # self-correlation to adjust prototype
            que_G, sup_G = que.transpose(1,2) @ que, sup.transpose(1,2) @ sup
            delta_G = que_G.unsqueeze(1) - sup_G.unsqueeze(0)
            selfcor = self.reweight(delta_G).squeeze() / (128. ** 0.5)
            proto_self = torch.sigmoid(selfcor) * new_proto
            
            # cross-correlation to adjust prototype
            que, sup = que.reshape(self.proj_dim, -1), sup.reshape(self.proj_dim, -1)
            crosscor = torch.matmul(que.transpose(0, 1) / (128. ** 0.5), sup)
            crosscor = crosscor.reshape(batch, dim, way, dim).permute(0, 2, 1, 3)
            crosscor = F.softmax(crosscor, dim=-1)
            proto_cross = torch.matmul(crosscor, new_proto.unsqueeze(2).transpose(-2, -1)).squeeze(-1)

            # integrate prototype and adjusted prototypes, 
            output = self.fc(proto_cross + proto_self)
            output = self.layer_norm(output + residual)
            proto = proto + output / kshot
            
        return proto

