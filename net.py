import torch
import torch.nn as nn

from pytorch3d.ops.knn import knn_gather, knn_points

import kit

class Network(nn.Module):
    def __init__(self, local_region, granularity, init_ratio, expand_ratio):
        super(Network, self).__init__()

        self.local_region = local_region
        self.init_ratio = init_ratio
        self.expand_ratio = expand_ratio
        self.granularity = granularity

        self.pt = kit.PT(in_channel=1, out_channel=128, n_layers=5)
        self.mu_sigma_pred = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 16), # 32 -> 16
            nn.ReLU(),
            nn.Linear(16, 1*2), #32 -> 16
        )

    def forward(self, batch_x):
        B, N, _ = batch_x.shape

        # random grouping
        base_size = min(N//self.init_ratio, self.granularity)
        window_size = base_size

        context_ls, target_ls = [], []
        cursor = base_size

        while cursor<N:
            window_size = min(window_size*self.expand_ratio, self.granularity)
            context_ls.append(batch_x[:, :cursor, :])
            target_ls.append(batch_x[:, cursor:cursor+window_size, :])
            cursor += window_size

        # attributes prediction
        total_bits = 0
        for i in range(len(context_ls)):
            target_geo, target_attr = target_ls[i][:, :, :3].clone(), target_ls[i][:, :, 3:].clone()
            context_geo, context_attr =  context_ls[i][:, :, :3].clone(), context_ls[i][:, :, 3:].clone()

            # context window gathering
            #opacity = 1, rotation = 360, scale = 10, sh_degree = ??
            
            min_opacity = context_attr[:, :, 0].min()
            max_opacity = context_attr[:, :, 0].max()
            # context_attr[:, :, 0] = (context_attr[:, :, 0] - min_opacity) / (max_opacity - min_opacity)

            # context_attr[:, :, 1:4] = context_attr[:, :, 1:4]
            _, idx, context_grouped_geo = knn_points(target_geo, context_geo, K=self.local_region, return_nn=True)
            context_grouped_attr = knn_gather(context_attr, idx)

            # spatial normalization
            # context_grouped_geo = context_grouped_geo - target_geo.view(B, -1, 1, 3)
            context_grouped_geo = kit.n_scale_ball(context_grouped_geo)
            
            # Network
            feature = self.pt(context_grouped_geo, context_grouped_attr)
            mu_sigma = self.mu_sigma_pred(feature)
            mu, sigma = mu_sigma[:, :, :1], torch.exp(mu_sigma[:, :, 1:])
            # print(f"mu: {mu}")
            # mu = torch.nan_to_num(mu, nan=0.0)
            # sigma = torch.nan_to_num(sigma, nan=0.0)  # 너무 작은 값이 되지 않도록 조정


            bits, _ = kit.feature_probs_based_mu_sigma(target_attr, mu, sigma)
            # bits, _ = kit.feature_probs_based_mu_sigma(target_attr, mu * (max_opacity - min_opacity) + min_opacity, sigma * (max_opacity - min_opacity))
            total_bits += bits
        
        return total_bits
