import os
import time
import argparse

import numpy as np

from glob import glob
from tqdm import tqdm

import torch
import torchac
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import kit
from net import Network
from scipy.spatial import KDTree  # KDTree를 이용해 순서 맞추기

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

# Argument parsing
parser = argparse.ArgumentParser(
    prog='gaussian_decompress.py',
    description='Decompress Point Cloud Opacity Attributes.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ckpt', required=True, help='Trained ckeckpoint file.')
parser.add_argument('--compressed_path', required=True, help='Compressed file saving directory.')
parser.add_argument('--decompressed_path', required=True, help='Decompressed file saving directory.')

parser.add_argument('--local_region', type=int, help='', default=8)
parser.add_argument('--granularity', type=int, help='', default=2**14)
parser.add_argument('--init_ratio', type=int, help='', default=1024)
parser.add_argument('--expand_ratio', type=int, help='', default=2)
parser.add_argument('--prg_seed', type=int, help='', default=2147483647)

args = parser.parse_args()

# Make sure the decompressed path exists
if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

comp_glob = os.path.join(args.compressed_path, '*.c.bin')
files = np.array(glob(comp_glob, recursive=True))
np.random.shuffle(files)
files = files[:]

# Load the network
net = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio, expand_ratio=args.expand_ratio)
net.load_state_dict(torch.load(args.ckpt))
# net = torch.compile(net, mode='max-autotune')
net.cuda().eval()

# Warm up model
_ = net.mu_sigma_pred(net.pt(torch.rand((1, 32, 8, 3)).cuda(), torch.rand((1, 32, 8, 1)).cuda()))

dec_times = []

with torch.no_grad():
    for comp_c_f in tqdm(files):
        fname = os.path.split(comp_c_f)[-1].split('.c.bin')[0]
        geo_f_path = os.path.join(args.compressed_path, fname+'.geo.bin')

        # Load original point cloud (before compression)
        batch_x_geo = torch.tensor(np.fromfile(geo_f_path, dtype=np.float32)).view(1, -1, 3)  # 원본 XYZ 좌표
        # original_xyz = batch_x_geo.cpu().numpy()[0]  # 원본 데이터 XYZ 좌표 저장

        context_attr_base = torch.tensor(np.fromfile(comp_c_f, dtype=np.float32)).view(1, -1, 1)

        torch.cuda.synchronize()
        TIME_STAMP = time.time()

        _, N, _ = batch_x_geo.shape

        base_size = min(N//args.init_ratio, args.granularity)
        window_size = base_size
        cursor = base_size
        i = 0
        while cursor < N:
            window_size = min(window_size*args.expand_ratio, args.granularity)

            context_geo = batch_x_geo[:, :cursor, :].cuda()
            if window_size >= args.granularity:
                context_geo = batch_x_geo[:, cursor:cursor+window_size, :].cuda()
            target_geo = batch_x_geo[:, cursor:cursor+window_size, :].cuda()
            cursor += window_size

            context_attr = context_attr_base.float().cuda()
            context_attr = context_attr.repeat((1, 1, 1))

            _, idx, context_grouped_geo = knn_points(target_geo, context_geo, K=net.local_region, return_nn=True)
            context_grouped_attr = knn_gather(context_attr, idx)

            context_grouped_geo = context_grouped_geo - target_geo.view(1, -1, 1, 3)
            context_grouped_geo = kit.n_scale_ball(context_grouped_geo)

            feature = net.pt(context_grouped_geo, context_grouped_attr)
            mu_sigma = net.mu_sigma_pred(feature)
            mu, sigma = mu_sigma[:, :, :1], torch.exp(mu_sigma[:, :, 1:])

            cdf = kit.get_cdf_reflactance(mu[0], sigma[0])
            cdf = cdf[:, 0, :]
            comp_f = os.path.join(args.compressed_path, fname+f'.{i}.bin')
            print(f"Completed: {comp_f}")
            with open(comp_f, 'rb') as fin:
                byte_stream = fin.read()
            decomp_attr = torchac.decode_int16_normalized_cdf(
                kit._convert_to_int_and_normalize(cdf, True).cpu(), byte_stream
            )
            decomp_attr = decomp_attr.view(1, -1, 1)
            decomp_attr = decomp_attr.float() / 16384.0
            context_attr_base = torch.cat((context_attr_base, decomp_attr), dim=1).clamp(0,1)
            i += 1
            

        decompressed_pc = torch.cat((batch_x_geo, context_attr_base), dim=-1)
        torch.cuda.synchronize()
        dec_times.append(time.time()-TIME_STAMP)
        decompressed_path = os.path.join(args.decompressed_path, fname+'.bin.ply')
        kit.save_point_cloud_reflactance(decompressed_pc[0].detach().cpu().numpy(), path=decompressed_path)


print('Max GPU Memory:', round(torch.cuda.max_memory_allocated(device=None)/1024/1024, 3), 'MB')
print('ave dec time:', round(np.array(dec_times).mean(), 3), 's')
