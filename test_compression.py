import kit
import torch
import torchac
from pytorch3d.ops.knn import knn_gather, knn_points


target_attr = torch.FloatTensor(1, 25, 1).uniform_(0, 1)
mu = target_attr.mean(dim=1, keepdim=True)
sigma = target_attr.std(dim=1, keepdim=True).clamp(1e-3, 1e1)
mu = mu.expand(-1, 25, -1)
sigma = sigma.expand(-1, 25, -1)


cdf = kit.get_cdf_reflactance(mu[0], sigma[0])

cdf_int = kit._convert_to_int_and_normalize(cdf, True).cpu()
cdf_diff = cdf_int[:,:, 1:] - cdf_int[:,:, :-1]
min_cdf_diff = cdf_diff.min()
print("Minimum cdf difference:", min_cdf_diff.item())

if min_cdf_diff.item() <= 0:
    raise ValueError("CDF is not strictly increasing.")

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