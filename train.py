import os
import argparse
import datetime

import numpy as np
from glob import glob

import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

import kit
from net import Network



torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)


parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training from scratch.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--training_data', required=True, help='Training data (Glob pattern).')
parser.add_argument('--model_save_folder', required=True, help='Directory where to save trained models.')

parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate every x steps.', default=30000)
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=150000)

parser.add_argument('--local_region', type=int, help='Neighbooring scope for context windows (i.e., K).', default=8)
parser.add_argument('--granularity', type=int, help='Upper limit for each group (i.e., s*).', default=2**16)
parser.add_argument('--init_ratio', type=int, help='The ratio for size of the very first group (i.e., alpha).', default=256)
parser.add_argument('--expand_ratio', type=int, help='Expand ratio (i.e., r)', default=2)

args = parser.parse_args()


def collate_fn(batch):
    max_points = max([sample.shape[0] for sample in batch])  # 가장 긴 샘플 길이

    padded_batch = []
    masks = []

    for sample in batch:
        num_points = sample.shape[0]

        # NumPy 배열을 텐서로 변환
        sample = torch.tensor(sample, dtype=torch.float32)  # 텐서로 변환

        # 패딩 추가
        padding = torch.zeros((max_points - num_points, sample.shape[1]))  # 0으로 패딩
        padded_sample = torch.cat([sample, padding], dim=0)  # 텐서를 이어붙임

        # 마스크 생성
        mask = torch.cat([torch.ones(num_points), torch.zeros(max_points - num_points)])  # 실제 데이터는 1, 패딩은 0

        padded_batch.append(padded_sample)
        masks.append(mask)

    padded_batch = torch.stack(padded_batch)  # 텐서로 변환 (batch_size, max_points, num_features)
    masks = torch.stack(masks)  # 마스크 텐서로 변환 (batch_size, max_points)

    return padded_batch, masks, max_points  # (batch_size, max_points, num_features), (batch_size, max_points)




# CREATE MODEL SAVE PATH
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

files = np.array(glob(args.training_data, recursive=True))
np.random.shuffle(files)
files = files[:]
# points = kit.read_point_clouds_ycocg(files)
points = kit.read_point_clouds_gaussian(files)
print(type(points))

loader = Data.DataLoader(
    dataset = points,
    batch_size = args.batch_size,
    shuffle = True,
    # collate_fn = collate_fn
)

ae = Network(local_region=args.local_region, granularity=args.granularity, init_ratio=args.init_ratio, expand_ratio=args.expand_ratio).cuda().train()
optimizer = torch.optim.Adam(ae.parameters(), lr=args.learning_rate)

bpps, losses = [], []
global_step = 0

for epoch in range(1, 9999):
    # print(datetime.datetime.now())
    # for step, (batch_x, mask, max_points) in enumerate(loader):
    for step, (batch_x) in enumerate(loader):
        B, N, _ = batch_x.shape
        # B, N = batch_x.shape
        batch_x = batch_x.cuda()
        # mask = mask.cuda()
        # print(B, N, max_points, mask.sum())
        

        optimizer.zero_grad()

        total_bits = ae(batch_x)
        # bpp = (total_bits * mask).sum() / mask.sum() / B
        bpp = total_bits / B / N
        loss = bpp

        loss.backward()

        optimizer.step()
        global_step += 1

        # PRINT
        losses.append(loss.item())
        bpps.append(bpp.item())

        if global_step % 500 == 0:
            print(f'Epoch:{epoch} | Step:{global_step} | bpp:{round(np.array(bpps).mean(), 5)} | Loss:{round(np.array(losses).mean(), 5)}')
            bpps, losses = [], []
        
         # LEARNING RATE DECAY
        if global_step % args.lr_decay_steps == 0:
            args.learning_rate = args.learning_rate * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.learning_rate}.')

        # SAVE MODEL
        if global_step % 500 == 0:
            torch.save(ae.state_dict(), args.model_save_folder + f'ckpt.pt')
        
        if global_step >= args.max_steps:
            break

    if global_step >= args.max_steps:
        break
