""" SegPN Network 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder_Seg
from models.quest import QUEST

class SegPN(nn.Module):
    def __init__(self, args):
        super(SegPN, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_points = args.pc_npts
        
        self.encoder = Encoder_Seg(input_points=2048, num_stages=3, embed_dim=60, k_neighbors=16, de_neighbors=10,
                                     alpha=1000, beta=30)
        
        self.Dim = 900
        self.bn = nn.Sequential(nn.BatchNorm1d(self.Dim),
                                nn.ReLU())
        self.fc = nn.Sequential(nn.Conv1d(self.Dim, 196, 1),
                                nn.BatchNorm1d(196),
                                nn.ReLU(), 
                                nn.Conv1d(196, 128, 1), 
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                )
        
        self.quest = QUEST()
        
    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        N_way, K_shot, C, PN = support_x.shape
        
        support_x = support_x.float().cuda().permute(0, 1, 3, 2).view(-1, PN, C)
        support_y = support_y.float().cuda().view(N_way, K_shot, PN)
        query_x = query_x.float().cuda().permute(0, 2, 1).view(-1, PN, C)
        query_y = query_y.cuda().view(-1, PN)
        
        # Pass through the Non-Parametric Encoder + Decoder
        with torch.no_grad():
            support_features = self.encoder(support_x, variant='training')
            support_features = support_features / support_features.norm(dim=1, keepdim=True)
            
            query_features = self.encoder(query_x, variant='training')
            query_features = query_features / query_features.norm(dim=1, keepdim=True)
        
        support_feat = self.bn(support_features)
        support_feat = self.fc(support_feat)
        support_feat = support_feat.permute(0, 2, 1)

        query_feat = self.bn(query_features)
        query_feat = self.fc(query_feat)
        query_feat = query_feat.permute(0, 2, 1)
        
        # obtain prototype
        feature_memory_list, label_memory_list = [], []
        support_feat = support_feat.view(N_way, K_shot, PN, -1)
        
        mask_bg = (support_y == 0)
        bg_features = support_feat[mask_bg]
        
        if bg_features.shape[0] < 1:
            bg_features = torch.ones(1, support_feat.shape[-1]).cuda() * 0.1
        else:
            bg_features = bg_features.mean(0).unsqueeze(0)
        feature_memory_list.append(bg_features)
        label_memory_list.append(torch.tensor(0).unsqueeze(0))
        for i in range(N_way):
            mask_fg = (support_y[i] == 1)
            fg_features = support_feat[i, mask_fg]
            fg_features = fg_features.mean(0).unsqueeze(0)
            feature_memory_list.append(fg_features)
            label_memory_list.append(torch.tensor(i+1).unsqueeze(0))

        feature_memory = torch.cat(feature_memory_list, dim=0)
        
        label_memory = torch.cat(label_memory_list, dim=0).cuda()
        label_memory = F.one_hot(label_memory, num_classes=N_way+1)

        feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)
        
        feature_memory = feature_memory.unsqueeze(0).repeat(N_way, 1, 1)
        feature_memory = self.quest(query_feat, support_feat, feature_memory)
        
        # similarity-based segmentation
        sim = [query_feat[i] @ feature_memory[i].t() for i in range(N_way)]
        sim = torch.stack(sim, dim=0)
        logits = sim @ label_memory.float()
        loss = F.cross_entropy(logits.reshape(-1, N_way+1), query_y.reshape(-1,).long())
        
        return logits, loss
