""" SegNN Network 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder_Seg

class SegNN(nn.Module):
    def __init__(self, args):
        super(SegNN, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dataset = args.dataset
        
        if args.dataset == 's3dis':
            self.encoder = Encoder_Seg(input_points=2048, num_stages=3, embed_dim=120, k_neighbors=16, de_neighbors=20,
                                        alpha=1000, beta=30)
        elif args.dataset == 'scannet':
            self.encoder = Encoder_Seg(input_points=2048, num_stages=2, embed_dim=120, k_neighbors=16, de_neighbors=20,
                                        alpha=1000, beta=20)
        self.encoder.eval()

    def forward(self, support_x, support_y, query_x, query_y):
        
        N_way, K_shot, C, PN = support_x.shape
        
        support_x = support_x.float().cuda().permute(0, 1, 3, 2).view(-1, PN, C)
        support_y = support_y.float().cuda().view(N_way, K_shot, PN)
        query_x = query_x.float().cuda().permute(0, 2, 1).view(-1, PN, C)
        query_y = query_y.cuda().view(-1, PN)
        
        # Pass through the Non-Parametric Encoder + Decoder
        with torch.no_grad():
            support_features, support_XYZ_features = self.encoder(support_x)
            support_features = support_features.permute(0, 2, 1)  # bz, 2048, c
            support_features = support_features / torch.norm(support_features, dim=-1, keepdim=True)
            
            support_XYZ_features = support_XYZ_features.permute(0, 2, 1)  # bz, 2048, c
            
            feature_memory_list, XYZ_memory_list = [], []
            label_memory_list = []
            support_features, support_XYZ_features = support_features.view(N_way, K_shot, PN, -1), support_XYZ_features.view(N_way, K_shot, PN, -1)
            for i in range(self.n_way):
                mask_fg = (support_y[i] == 1)

                fg_features = support_features[i, mask_fg]
                fg_features = fg_features.mean(0).unsqueeze(0)
                feature_memory_list.append(fg_features)
                label_memory_list.append(torch.tensor(i+1).unsqueeze(0))
                
                fg_XYZ_features = support_XYZ_features[i, mask_fg]
                fg_XYZ_features = fg_XYZ_features.mean(0).unsqueeze(0)
                XYZ_memory_list.append(fg_XYZ_features)

            # Find the point indices for the part_label within a shape
            mask_bg = (support_y == 0)
            bg_features = support_features[mask_bg]
            bg_features = bg_features.mean(0).unsqueeze(0)
            feature_memory_list.append(bg_features)
            label_memory_list.append(torch.tensor(0).unsqueeze(0))
        
            bg_XYZ_features = support_XYZ_features[mask_bg]
            bg_XYZ_features = bg_XYZ_features.mean(0).unsqueeze(0)
            XYZ_memory_list.append(bg_XYZ_features)

            feature_memory = torch.cat(feature_memory_list, dim=0)
            XYZ_memory = torch.cat(XYZ_memory_list, dim=0)
            
            label_memory = torch.cat(label_memory_list, dim=0).cuda()
            label_memory = F.one_hot(label_memory, num_classes=self.n_way+1)

            feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)
            feature_memory = feature_memory.permute(1, 0)
            XYZ_memory = XYZ_memory / torch.norm(XYZ_memory, dim=-1, keepdim=True)
            XYZ_memory = XYZ_memory.permute(1, 0)
            
            query_features, query_XYZ_features = self.encoder(query_x)
            query_features = query_features.permute(0, 2, 1)  # bz, 2048, c
            query_features /= query_features.norm(dim=-1, keepdim=True)
            
            query_XYZ_features = query_XYZ_features.permute(0, 2, 1)  # bz, 2048, c
            query_XYZ_features /= query_XYZ_features.norm(dim=-1, keepdim=True)
        
            Sim = query_features @ feature_memory
            Sim_XYZ = query_XYZ_features @ XYZ_memory
            
            if self.dataset == 's3dis':
                logits = (-100 * (1 - Sim)).exp() @ label_memory.float()
                logits_XYZ = (-100 * (1 - Sim_XYZ)).exp() @ label_memory.float()
            elif self.dataset == 'scannet':
                logits = (-100 * (1 - Sim)).exp() @ label_memory.float()
                logits_XYZ = (-100 * (1 - Sim_XYZ)).exp() @ label_memory.float()
            logits = logits + logits_XYZ
            
            logits = F.softmax(logits, dim=-1)
                        
        return logits, 0
    

    
    