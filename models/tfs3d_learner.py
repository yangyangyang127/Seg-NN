""" Learner for Few-shot 3D Point Cloud Semantic Segmentation
"""
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

from models.tfs3d import TFS3D
from models.tfs3d_t import TFS3D_T


class Learner(object):
    def __init__(self, args):

        # init model and optimizer
        if args.model == 'training_free':
            self.model = TFS3D(args)
        elif args.model == 'training':
            self.model = TFS3D_T(args)
        
        if torch.cuda.is_available():
            self.model.cuda()

        if args.model == 'training':
            
            print('setting optimizer : ')
            self.optimizer = torch.optim.AdamW(
            [{'params': self.model.fc.parameters()},
            {'params': self.model.att_learner.parameters()},
            {'params': self.model.bn1.parameters()}], lr=args.lr, weight_decay=0.1)
                
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        
        elif args.model == 'training_free':
            pass

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, support_y, query_x, query_y] = data
        self.model.train()

        query_logits, loss = self.model(support_x, support_y, query_x, query_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        query_pred = F.softmax(query_logits, dim=-1).argmax(dim=-1)
        
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return loss, accuracy

    def test(self, data, sampled_classes):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        self.model.eval()

        with torch.no_grad():
            logits, _ = self.model(support_x, support_y, query_x, query_y)
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)

            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return pred, accuracy
    
    