import kit
import torch
import torchac
from pytorch3d.ops.knn import knn_gather, knn_points
from torch.distributions.laplace import Laplace
def get_cdf_reflactance(mu, sigma):
    M, d = sigma.shape
    
    # mu와 sigma를 반복적으로 확장 (20000개의 flag에 대해 계산)
    mu = mu.unsqueeze(-1).repeat(1, 1, 17000)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, 17000)

    # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    gaussian = Laplace(mu, sigma)
    

    flag_min = 0.0
    flag_max = 1
    flags = torch.linspace(flag_min, flag_max, 17000).to(sigma.device)
    flags = flags.view(1, 1, -1).repeat(M, d, 1)

    # Laplace 분포 객체 생성
    laplace_dist = Laplace(mu, sigma)
    flag_min = torch.tensor(0.0).to(mu.device)
    flag_max = torch.tensor(1).to(mu.device)
    total_prob = laplace_dist.cdf(flag_max) - laplace_dist.cdf(flag_min)
    cdf = (laplace_dist.cdf(flags) - laplace_dist.cdf(flag_min)) / total_prob
    

    # CDF에 0 값을 추가
    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    return cdf_with_0

target_attr = torch.FloatTensor(1, 25, 1).uniform_(0, 1)
mu = target_attr.mean(dim=1, keepdim=True)
sigma = target_attr.std(dim=1, keepdim=True).clamp(1e-3, 1e1)
mu = torch.FloatTensor(1, 25, 1).uniform_(1, 2)
sigma = torch.FloatTensor(1, 25, 1).uniform_(0, 1)
mu = mu.expand(-1, 25, -1)
sigma = sigma.expand(-1, 25, -1)


cdf = get_cdf_reflactance(mu[0], sigma[0])

cdf_int = kit._convert_to_int_and_normalize(cdf, True).cpu()
cdf_diff = cdf_int[:,:, 1:] - cdf_int[:,:, :-1]
min_cdf_diff = cdf_diff.min()
print("Minimum cdf difference:", min_cdf_diff.item())

# if min_cdf_diff.item() <= 0:
#     raise ValueError("CDF is not strictly increasing.")

target_feature = (target_attr[0] * 16384).to(torch.int16)
cdf = cdf[:, 0, :]
target_feature = target_feature[:, 0]


byte_stream = torchac.encode_int16_normalized_cdf(
    kit._convert_to_int_and_normalize(cdf, True).cpu(),
    target_feature.cpu()
)


decomp_attr = torchac.decode_int16_normalized_cdf(
    kit._convert_to_int_and_normalize(cdf, True).cpu(), byte_stream
)
decomp_attr = decomp_attr.view(1, -1, 1)
decomp_attr = decomp_attr.float() / 16384.0


for original, decompressed in zip(target_attr[0].numpy().flatten(), decomp_attr[0].numpy().flatten()):
    print(f"Original: {original:.4f}, Decompressed: {decompressed:.4f}")