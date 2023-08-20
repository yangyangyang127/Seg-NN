# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import math
import torch
import torch.nn as nn
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

from models.model_utils import *


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x, rgb, rgbx):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)
        
        lc_rgb = index_points(rgb, fps_idx)
        lc_rgbx = index_points(rgbx, fps_idx)
        
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        
        knn_rgb = index_points(rgb, knn_idx)
        knn_rgbx = index_points(rgbx, knn_idx)

        return lc_xyz, lc_x, lc_rgb, lc_rgbx, knn_xyz, knn_x, knn_rgb, knn_rgbx


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, B_xyz, B_rgb):
        super().__init__()
        alpha, beta = 1, 1
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta, B_xyz, B_rgb)

    def forward(self, lc_xyz, lc_x, lc_rgb, lc_rgbx, knn_xyz, knn_x, knn_rgb, knn_rgbx):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        
        mean_rgb = lc_rgb.unsqueeze(dim=-2)
        std_rgb = torch.std(knn_rgb - mean_rgb)
        mean_rgbx = lc_rgbx.unsqueeze(dim=-2)
        std_rgbx = torch.std(knn_rgbx - mean_rgbx)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)
        knn_rgb = (knn_rgb - mean_rgb) / (std_rgb + 1e-5)
        knn_rgbx = (knn_rgbx - mean_rgbx) / (std_rgbx + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)
        knn_rgbx = torch.cat([knn_rgbx, lc_rgbx.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_rgb = knn_rgb.permute(0, 3, 1, 2)
        knn_rgbx = knn_rgbx.permute(0, 3, 1, 2)
        
        knn_x_w, knn_rgbx_w = self.geo_extract(knn_xyz, knn_x, knn_rgb)

        return knn_x_w, knn_rgbx_w

class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w, knn_rgbx_w):
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        
        lc_rgbx = knn_rgbx_w.max(-1)[0] + knn_rgbx_w.mean(-1)
        lc_rgbx = self.out_transform(lc_rgbx)
        
        return lc_x, lc_rgbx


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, x, rgbx):
        B, _, N = x.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda() / feat_dim   
        dim_embed = torch.pow(self.alpha, feat_range)
        x_div = torch.div(self.beta * x.unsqueeze(-1), dim_embed)
        rgbx_div = torch.div(self.beta * rgbx.unsqueeze(-1), dim_embed)
        
        sin_x, cos_x = torch.sin(x_div), torch.cos(x_div)
        sin_rgbx, cos_rgbx = torch.sin(rgbx_div), torch.cos(rgbx_div)
        
        position_embed = torch.stack([sin_x, cos_x], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        rgbx_embed = torch.stack([sin_rgbx, cos_rgbx], dim=4).flatten(3)
        rgbx_embed = rgbx_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed, rgbx_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta, B_xyz, B_rgb):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.B_xyz, self.B_rgb = B_xyz, B_rgb
        
    def forward(self, knn_xyz, knn_x, knn_rgb):

        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        knn_xyz_1 = knn_xyz.permute(0, 2, 3, 1)[..., None]
        div_xyz = torch.matmul(2 * math.pi * knn_xyz_1, self.B_xyz[:, :feat_dim].cuda())
        sin_xyz, cos_xyz = torch.sin(div_xyz), torch.cos(div_xyz)
        xyz_embed = torch.cat([sin_xyz, cos_xyz], dim=4).flatten(3)
        xyz_embed = xyz_embed.permute(0, 3, 1, 2)
        
        knn_rgb_1 = knn_rgb.permute(0, 2, 3, 1)[..., None]
        div_rgb = torch.matmul(2 * math.pi * knn_rgb_1, self.B_rgb[:, :feat_dim].cuda())
        sin_rgb, cos_rgb = torch.sin(div_rgb), torch.cos(div_rgb)
        rgb_embed = torch.cat([sin_rgb, cos_rgb], dim=4).flatten(3)
        rgb_embed = rgb_embed.permute(0, 3, 1, 2)

        # Weigh 
        knn_x_w = (knn_x + xyz_embed) * xyz_embed
        knn_rgbx_w = (knn_x + xyz_embed) * rgb_embed

        knn_x_w = 0.8*knn_x_w + 0.2*knn_rgbx_w

        return knn_x_w, knn_rgbx_w


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, B_xyz, B_rgb):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, alpha, beta)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            if i < 2:
                out_dim = out_dim * 2
                group_num = group_num // 2
            else:
                out_dim = out_dim * 2
                group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, B_xyz, B_rgb))
            self.Pooling_list.append(Pooling(out_dim))
            

    def forward(self, xyz, x, rgb, rgbx):

        # Raw-point Embedding
        x, rgbx = self.raw_point_embed(x, rgbx)

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]
                
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, rgb, lc_rgbx, knn_xyz, knn_x, knn_rgb, knn_rgbx \
                = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1), rgb, rgbx.permute(0,2,1))
            # Local Geometry Aggregation
            knn_x_w, knn_rgbx_w = self.LGA_list[i](xyz, lc_x,  rgb, lc_rgbx, \
                                                    knn_xyz, knn_x, knn_rgb, knn_rgbx)
            # Pooling
            x, rgbx = self.Pooling_list[i](knn_x_w, knn_rgbx_w)

            xyz_list.append(xyz)
            x_list.append(x)

        return xyz_list, x_list


# Non-Parametric Decoder
class DecNP(nn.Module):  
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors

    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            #dists = dists ** 2
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]
            
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
                        
            weight = weight.view(B, N, self.de_neighbors, 1)
            points2_indexed = index_points(points2, idx)

            interpolated_points = torch.sum(points2_indexed * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points


    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x


# Non-Parametric Network
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=144, k_neighbors=128, de_neighbors=6, alpha=1000, beta=50):
        super().__init__()
        
        B_xyz = torch.abs(torch.randn(1, 5000)) * 0.1
        B_rgb = torch.abs(torch.randn(1, 5000)) * 0.1
        self.EncNP_1 = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta, B_xyz, B_rgb)
        
        B_xyz = torch.abs(torch.randn(1, 5000)) * 0.1
        B_rgb = torch.abs(torch.randn(1, 5000)) * 0.1
        self.EncNP_2 = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta, B_xyz, B_rgb)
        
        self.DecNP = DecNP(num_stages, de_neighbors)
        
    def forward(self, x, variant='training_free'):

        pos, rgb = x[..., 6:], x[..., 3:6]
        xyz, pos_x = pos, pos.permute(0, 2, 1)
        rgb, rgbx = rgb, rgb.permute(0, 2, 1)

        xyz_list, x_list = self.EncNP_1(xyz, pos_x, rgb, rgbx)
        xyz_feat = self.DecNP(xyz_list, x_list)
        
        if variant == 'training':
            return xyz_feat
        
        pos, rgb = x[..., 0:3], x[..., 3:6]
        XYZ, pos_X = pos, pos.permute(0, 2, 1)
        rgb, rgbx = rgb, rgb.permute(0, 2, 1)
        
        XYZ_list, X_list = self.EncNP_2(XYZ, pos_X, rgb, rgbx)
        XYZ_feat = self.DecNP(XYZ_list, X_list)
        
        return xyz_feat, XYZ_feat