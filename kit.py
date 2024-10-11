import os
import math
import multiprocessing
from torch.distributions import Beta
from scipy.special import betainc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm import tqdm
from plyfile import PlyElement, PlyData
from pyntcloud import PyntCloud
from pytorch3d.ops.knn import knn_gather, knn_points
from torch.distributions.laplace import Laplace


#core transformation function
def transformRGBToYCoCg(bitdepth, rgb):
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    co = r - b
    t = b + (co >> 1) # co >>1 i.e. co // 2
    cg = g - t
    y = t + (cg >> 1)

    offset = 1 << bitdepth # 2^bitdepth

    # NB: YCgCoR needs extra 1-bit for chroma
    return np.column_stack((y,  co + offset, cg + offset))


def transformYCoCgToRGB(bitDepth,  ycocg):

    offset = 1 << bitDepth
    y0 = ycocg[:,0]
    co = ycocg[:,1] - offset
    cg = ycocg[:,2] - offset

    t = y0 - (cg >> 1)

    g = cg + t
    b = t - (co >> 1)
    r = co + b

    maxVal = (1 << bitDepth) - 1
    r = np.clip(r, 0, maxVal)
    g = np.clip(g, 0, maxVal)
    b = np.clip(b, 0, maxVal)

    return np.column_stack((r,g,b))


def read_point_cloud_ycocg(filepath):
    pc = PyntCloud.from_file(filepath)
    try:
        cols=['x', 'y', 'z','red', 'green', 'blue']
        points=pc.points[cols].values
    except:
        cols = ['x', 'y', 'z', 'r', 'g', 'b']
        points = pc.points[cols].values
    color = points[:, 3:].astype(np.int16)
    color = transformRGBToYCoCg(8, color)
    # color: int
    # y channel: 0~255
    # co channel: 0~511 (1~511 in our dataset)
    # cg channel: 0~511 (34~476 in our dataset)
    points[:, 3:] = color.astype(float)
    return points

def read_point_cloud_gaussian(filepath):
    pc = PyntCloud.from_file(filepath)
    cols=['x', 'y', 'z', 'opacity']
    # cols=['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2','f_dc_0', 'f_dc_1','f_dc_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    points=pc.points[cols].values
    return points


def save_point_cloud_ycocg(pc, path):
    color = pc[:, 3:]
    color = np.round(color).astype(np.int16) # 务必 round 后 再加 astype
    color = transformYCoCgToRGB(8, color)

    pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    pc[['red','green','blue']] = np.round(color).astype(np.uint8)
    cloud = PyntCloud(pc)
    cloud.to_file(path)

def read_point_cloud_reflactance(filepath):
    plydata = PlyData.read(filepath)
    pc = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'], plydata['vertex']['reflectance'])))).astype(np.float32)        
    return pc

def read_point_cloud_gaussian_att(filepath):
    plydata = PlyData.read(filepath)
    pc = np.array(np.transpose(np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
                                         plydata['vertex']['opacity'], plydata['vertex']['scale_0'],
                                         plydata['vertex']['scale_1'], plydata['vertex']['scale_2'],
                                         plydata['vertex']['f_dc_0'], plydata['vertex']['f_dc_1'],
                                         plydata['vertex']['f_dc_2'],
                                         plydata['vertex']['rot_0'], plydata['vertex']['rot_1'],
                                         plydata['vertex']['rot_2'], plydata['vertex']['rot_3'])))).astype(np.float32)
    return pc

def read_point_cloud_gaussian_opacity(filepath):
    plydata = PlyData.read(filepath)
    pc = np.array(np.transpose(np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
                                         plydata['vertex']['opacity'])))).astype(np.float32)
    return pc



def save_point_cloud_reflactance(pc, path, to_rgb=False):

    if to_rgb:
        pc[:, 3:] = pc[:, 3:] / 100
        cmap = plt.get_cmap('jet')
        color = np.round(cmap(pc[:, 3])[:, :3] * 255)
        pc = np.hstack((pc[:, :3], color))
        pc = pd.DataFrame(pc, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        pc[['red','green','blue']] = np.round(np.clip(pc[['red','green','blue']], 0, 255)).astype(np.uint8)
        cloud = PyntCloud(pc)
        cloud.to_file(path)
    else:
        scan = pc
        vertex = np.array(
            [(scan[i,0], scan[i,1], scan[i,2], scan[i,3]) for i in range(scan.shape[0])],
            dtype=[
                ("x", np.dtype("float32")), 
                ("y", np.dtype("float32")), 
                ("z", np.dtype("float32")),
                ("opacity", np.dtype("float32")),
            ]
        )
        PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        output_pc = PlyElement.describe(vertex, "vertex")
        output_pc = PlyData([output_pc])
        output_pc.write(path)


def read_point_clouds_ycocg(file_path_list, bar=True):
    print('loading point clouds...')
    with multiprocessing.Pool() as p:
        if bar:
            pcs = list(tqdm(p.imap(read_point_cloud_ycocg, file_path_list, 32), total=len(file_path_list)))
        else:
            pcs = list(p.imap(read_point_cloud_ycocg, file_path_list, 32))
    return pcs


def read_point_clouds_gaussian(file_path_list, bar=True):
    print('loading point clouds...')
    with multiprocessing.Pool() as p:
        if bar:
            pcs = list(tqdm(p.imap(read_point_cloud_gaussian, file_path_list, 32), total=len(file_path_list)))
        else:
            pcs = list(p.imap(read_point_cloud_gaussian, file_path_list, 32))
    return pcs


def n_scale_ball(grouped_xyz):
    B, N, K, _ = grouped_xyz.shape

    longest = (grouped_xyz**2).sum(dim=-1).sqrt().max(dim=-1)[0]
    scaling = (1) / longest
    
    grouped_xyz = grouped_xyz * scaling.view(B, N, 1, 1)

    return grouped_xyz


class MLP(nn.Module):
    def __init__(self, in_channel, mlp, relu, bn):
        super(MLP, self).__init__()

        mlp.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlp) - 1):
            if relu[i]:
                if bn[i]:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.BatchNorm2d(mlp[i+1]),
                        nn.ReLU(),
                    )
                else:
                    mlp_Module = nn.Sequential(
                        nn.Conv2d(mlp[i], mlp[i+1], 1),
                        nn.ReLU(),
                    )
            else:
                mlp_Module = nn.Sequential(
                    nn.Conv2d(mlp[i], mlp[i+1], 1),
                )
            self.mlp_Modules.append(mlp_Module)


    def forward(self, points, squeeze=False):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D, N]
        """
        if squeeze:
            points = points.unsqueeze(-1) # [B, C, N, 1]
        
        
        for m in self.mlp_Modules:
            points = m(points)
            
        # [B, D, N, 1]
        
        if squeeze:
            points = points.squeeze(-1) # [B, D, N] 

        return points
    

class QueryMaskedAttention(nn.Module):
    def __init__(self, channel):
        super(QueryMaskedAttention, self).__init__()
        self.channel = channel
        self.k_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.v_mlp = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.pe_multiplier, self.pe_bias = True, True
        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=channel, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.weight_encoding = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        )
        self.residual_emb = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, grouped_xyz, grouped_feature):

        key = self.k_mlp(grouped_feature) # B, C, K, M
        value = self.v_mlp(grouped_feature) # B, C, K, M

        relation_qk = key #  - query
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(grouped_xyz)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(grouped_xyz)
            relation_qk = relation_qk + peb
            value = value + peb

        weight  = self.weight_encoding(relation_qk)
        score = self.softmax(weight) # B, C, K, M

        feature = score*value # B, C, K, M
        feature = self.residual_emb(feature) # B, C, K, M

        return feature


class PT(nn.Module):
    def __init__(self, in_channel, out_channel, n_layers):
        super(PT, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_layers = n_layers
        self.sa_ls, self.sa_emb_ls = nn.ModuleList(), nn.ModuleList()
        self.linear_in = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        for i in range(n_layers):
            self.sa_emb_ls.append(nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1),
                nn.ReLU(),
            ))
            self.sa_ls.append(QueryMaskedAttention(out_channel))
    def forward(self, groped_geo, grouped_attr):
        """
        Input:
            groped_geo: input points position data, [B, M, K, 3]
            groped_attr: input points feature data, [B, M, K, 3]
        Return:
            feature: output feature data, [B, M, C]
        """
        groped_geo, grouped_attr = groped_geo.permute((0, 3, 2, 1)), grouped_attr.permute((0, 3, 2, 1)) # B, _, K, M
        feature = self.linear_in(grouped_attr)
        for i in range(self.n_layers):
            identity = feature
            feature = self.sa_emb_ls[i](feature)
            output = self.sa_ls[i](groped_geo, feature)
            feature = output + identity
        feature = feature.sum(dim=2).transpose(1, 2)
        return feature


def get_cdf(mu, sigma):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, 256)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 256).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, 256).to(sigma.device).view(1, 1, 256).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    return cdf_with_0


def get_cdf_ycocg(mu, sigma):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, 512)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 512).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, 512).to(sigma.device).view(1, 1, 512).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    return cdf_with_0


def get_cdf_reflactance(mu, sigma):
    M, d = sigma.shape
    
    # mu와 sigma를 반복적으로 확장 (20000개의 flag에 대해 계산)
    mu = mu.unsqueeze(-1).repeat(1, 1, 17000)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 17000)

    # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    gaussian = Laplace(mu, sigma)
    
    # mu와 sigma의 최소, 최대 값 계산
    mu_min, mu_max = mu.min(), mu.max()
    sigma_mean = sigma.mean()

    flag_min = 0.0
    flag_max = 1.0
    flags = torch.linspace(flag_min, flag_max, 17000).to(sigma.device)
    flags = flags.view(1, 1, -1).repeat(M, d, 1)

    # Laplace 분포 객체 생성
    laplace_dist = Laplace(mu, sigma)
    flag_min = torch.tensor(0.0).to(mu.device)
    flag_max = torch.tensor(1.0).to(mu.device)
    total_prob = laplace_dist.cdf(flag_max) - laplace_dist.cdf(flag_min)
    cdf = (laplace_dist.cdf(flags) - laplace_dist.cdf(flag_min)) / total_prob
    

    # CDF에 0 값을 추가
    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    
    return cdf_with_0

def feature_probs_based_mu_sigma(feature, mu, sigma):
    sigma = sigma.clamp(1e-5, 1e10)
    # print(mu.shape, sigma.shape, feature.shape)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(feature+0.005) - gaussian.cdf(feature-0.005)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, probs

def get_cdf_reflectance_beta(alpha, beta):
    M, d = alpha.shape

    # 플래그 범위 설정
    flag_min = 0
    flag_max = 1
    num_flags = 17000
    flags = torch.linspace(flag_min, flag_max, num_flags).to(alpha.device)
    flags = flags.view(1, 1, -1).repeat(M, d, 1)

    # alpha와 beta를 확장
    alpha_expanded = alpha.unsqueeze(-1).repeat(1, 1, num_flags)
    beta_expanded = beta.unsqueeze(-1).repeat(1, 1, num_flags)


    cdf = torch.tensor(betainc(alpha_expanded.cpu().numpy(), beta_expanded.cpu().numpy(), flags.cpu().numpy())).to(alpha.device)

    return cdf

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(65536, dtype=torch.float32, device=cdf_float.device)  # 2**16 사용
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value).round()
    
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.float32, device=cdf_float.device)
        cdf_float.add_(r)
    
    cdf_float = torch.clip(cdf_float, 0, 65535)
    return cdf_float.to(torch.int16, non_blocking=True)