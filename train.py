import os
import argparse
import datetime
import numpy as np
from glob import glob

import torch
import torch.utils.data as Data

import kit
from net import Network
import wandb

# Initialize wandb
wandb.init(project='pcac', entity='ownvoy')
wandb.config.learning_rate = 0.0001
wandb.config.batch_size = 1
wandb.config.lr_decay = 0.1
wandb.config.lr_decay_steps = 30000
wandb.config.max_steps = 170000
wandb.config.local_region = 8
wandb.config.granularity = 2**14
wandb.config.init_ratio = 256
wandb.config.expand_ratio = 2

# Seed setting
torch.cuda.manual_seed(11)
torch.manual_seed(11)
np.random.seed(11)

# Argument parser setup
parser = argparse.ArgumentParser(prog='train.py', description='Training from scratch.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--training_data', required=True, help='Training data (Glob pattern).')
parser.add_argument('--model_save_folder', required=True, help='Directory where to save trained models.')
args = parser.parse_args()

# Directory setup for model saving
if not os.path.exists(args.model_save_folder):
    os.makedirs(args.model_save_folder)

# Data preparation
files = np.array(glob(args.training_data, recursive=True))
np.random.shuffle(files)
points = kit.read_point_clouds_gaussian(files)

# DataLoader setup
loader = Data.DataLoader(dataset=points, batch_size=wandb.config.batch_size, shuffle=True)

# Network and optimizer setup
ae = Network(local_region=wandb.config.local_region, granularity=wandb.config.granularity, init_ratio=wandb.config.init_ratio, expand_ratio=wandb.config.expand_ratio).cuda().train()
optimizer = torch.optim.Adam(ae.parameters(), lr=wandb.config.learning_rate)

# Training loop
global_step = 0
min_bpp = 1000000
for epoch in range(1, 9999):
    epoch_losses = []
    epoch_bpps = []

    for step, (batch_x) in enumerate(loader):
        batch_x = batch_x.cuda()
        optimizer.zero_grad()
        total_bits = ae(batch_x)
        bpp = total_bits / wandb.config.batch_size / batch_x.shape[1]
        loss = bpp
        bpp = bpp.item()
        loss.backward()
        optimizer.step()
        global_step += 1

        # Accumulate metrics
        epoch_losses.append(loss.item())
        epoch_bpps.append(bpp)

        if global_step % 200 == 0:
            print(f"Global Step: {global_step}, BPP: {bpp}")

        # Model saving condition
        

        # Learning rate decay
        if global_step % wandb.config.lr_decay_steps == 0:
            new_lr = optimizer.param_groups[0]['lr'] * wandb.config.lr_decay
            optimizer.param_groups[0]['lr'] = new_lr
            print(f'Learning rate decay triggered at step {global_step}, LR is now {new_lr}.')

        if global_step >= wandb.config.max_steps:
            break

    # Log epoch metrics
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    avg_bpp = sum(epoch_bpps) / len(epoch_bpps)
    wandb.log({"epoch": epoch, "avg_loss": avg_loss, "avg_bpp": avg_bpp, "learning_rate": optimizer.param_groups[0]['lr']})
    if min_bpp >= avg_bpp:
            min_bpp = avg_bpp
            print("model saved")
            torch.save(ae.state_dict(), os.path.join(args.model_save_folder, 'ckpt.pt'))
            wandb.save(os.path.join(args.model_save_folder, 'ckpt.pt'))
    if global_step >= wandb.config.max_steps:
        break
