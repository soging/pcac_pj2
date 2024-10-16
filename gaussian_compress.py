import os
import time
import argparse

import numpy as np

from glob import glob
from tqdm import tqdm

import torch
import torchac
from pytorch3d.ops.knn import knn_gather, knn_points

import kit
from net import Network

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(
    prog='gaussian_compress.py',
    description='Compress Point Cloud Gaussian Attributes.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ckpt', required=True, help='Trained checkpoint file.')
parser.add_argument('--input_glob', required=True, help='Point clouds glob pattern to be compressed.')
parser.add_argument('--compressed_path', required=True, help='Compressed file saving directory.')

parser.add_argument('--local_region', type=int, help='Neighboring scope for context windows (i.e., K).', default=8)
parser.add_argument('--granularity', type=int, help='Upper limit for each group (i.e., s*).', default=2 ** 16)
parser.add_argument('--init_ratio', type=int, help='The ratio for size of the very first group (i.e., alpha).',
                    default=256)
parser.add_argument('--expand_ratio', type=int, help='Expand ratio (i.e., r)', default=2)
parser.add_argument('--prg_seed', type=int, help='Pseudorandom seed for PRG.', default=2147483647)

args = parser.parse_args()

if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)

files = np.array(glob(args.input_glob, recursive=True))
files = files[:]

net = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio,
              expand_ratio=args.expand_ratio)
net.load_state_dict(torch.load(args.ckpt))
net = torch.compile(net, mode='max-autotune')
net.cuda().eval()

# Warm up the model (for better performance)
_ = net.mu_sigma_pred(net.pt(torch.rand((1, 32, 8, 3)).cuda(), torch.rand((1, 32, 8, 1)).cuda()))

enc_times = []
fnames, bpps = [], []
with torch.no_grad():
    for f in tqdm(files):
        fname = os.path.split(f)[-1]

        pc = kit.read_point_cloud_gaussian_opacity(f)
        batch_x = torch.tensor(pc).unsqueeze(0)
        batch_x = batch_x.cuda()

        B, N, _ = batch_x.shape
        print(B,N)

        torch.cuda.synchronize()
        TIME_STAMP = time.time()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(args.prg_seed)

        base_size = min(N // args.init_ratio, args.granularity)
        window_size = base_size
        context_ls, target_ls = [], []
        cursor = base_size

        # Debug: Total points in the cloud
        print(f"Total Points: {N}")

        while cursor < N:
            window_size = min(window_size * args.expand_ratio, args.granularity)
            
            # Debug: 현재 cursor와 window_size 값 출력
            # print(f'Current Cursor: {cursor}, Window Size: {window_size}')
            
            context_ls.append(batch_x[:, :cursor, :])
            target_ls.append(batch_x[:, cursor:cursor + window_size, :])
            cursor += window_size

        # # 마지막으로 남은 포인트 처리
        # if cursor < N:
        #     print(f'Processing remaining points from {cursor} to {N}')
        #     context_ls.append(batch_x[:, :cursor, :])
        #     target_ls.append(batch_x[:, cursor:, :])

        total_ac_time = 0
        total_bits = 0
        for i in range(len(target_ls)):
            target_geo, target_attr = target_ls[i][:, :, :3].clone(), target_ls[i][:, :, 3:].clone()
            context_geo, context_attr = context_ls[i][:, :, :3].clone(), context_ls[i][:, :, 3:].clone()
            target_attr, context_attr = target_attr.repeat((1, 1, 1)), context_attr.repeat((1, 1, 1))

            context_attr[:, :, 0] = context_attr[:, :, 0] / 1000  # Opacity normalization
            # Other normalizations can be applied here

            _, idx, context_grouped_geo = knn_points(target_geo, context_geo, K=net.local_region, return_nn=True)
            context_grouped_attr = knn_gather(context_attr, idx)

            context_grouped_geo = context_grouped_geo - target_geo.view(B, -1, 1, 3)
            context_grouped_geo = kit.n_scale_ball(context_grouped_geo)

            feature = net.pt(context_grouped_geo, context_grouped_attr)
            mu_sigma = net.mu_sigma_pred(feature)
            mu, sigma = mu_sigma[:, :, :1]+0.5, torch.exp(mu_sigma[:, :, 1:])

            cdf = kit.get_cdf_reflactance(mu[0]*1000, sigma[0]*32)
            target_feature = (target_attr[0]).to(torch.int16)
            cdf = cdf[:, 0, :]
            target_feature = target_feature[:, 0]

            # byte_stream = torchac.encode_float_cdf(cdf.cpu(), target_feature.cpu(), check_input_bounds=True)
            
            byte_stream = torchac.encode_int16_normalized_cdf(
                kit._convert_to_int_and_normalize(cdf, True).cpu(),
                target_feature.cpu())

            comp_f = os.path.join(args.compressed_path, fname + f'.{i}.bin')
            with open(comp_f, 'wb') as fout:
                fout.write(byte_stream)

            total_bits += kit.get_file_size_in_bits(comp_f)

        comp_base_f = os.path.join(args.compressed_path, fname + '.c.bin')
        context_base = context_ls[0][0, :, 3:].detach().cpu().numpy()

        torch.cuda.synchronize()
        enc_times.append(time.time() - TIME_STAMP)

        context_base.astype(np.float32).tofile(comp_base_f)
        total_bits += kit.get_file_size_in_bits(comp_base_f)

        geo_f = os.path.join(args.compressed_path, fname + '.geo.bin')
        batch_x[:, :, :3].detach().cpu().numpy().astype(np.float32).tofile(geo_f)

        fnames.append(fname)
        bpps.append(np.round(total_bits / N, 3))

print('Memory:', round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3), 'MB')
print(f'Done! Total {len(fnames)} | opacity bpp: {round(np.array(bpps).mean(), 3)} | ave enc time: {round(np.array(enc_times).mean(), 3)} s')
print('Params:', sum(p.numel() for p in net.parameters()), 'Trainable params:', sum(p.numel() for p in net.parameters() if p.requires_grad))
