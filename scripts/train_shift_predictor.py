# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import copy

import subprocess

from urllib.request import urlopen

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torchvision import datasets
from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder

import clip

from torchvision.models import ResNet50_Weights


from s3c.models.heads import  ShiftPredictor2 #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset

import timm

from PIL import Image

from tqdm import tqdm

from datetime import datetime


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256 #
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768

epoch_teacher = 20

n_saccades_max = 30 

zoom = 1.5
std = 0.5 / zoom 

n_sab = 4 #

n_heads = 12

orig = False
grid = False

train_epochs = 100

alpha = 3e-6

stop_gradient = False

suffix = ""
suffix = suffix + f"_a{alpha}"
    
if grid : suffix = suffix + "_GRID"
if orig: suffix = suffix + "_ORIG"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_SHIFT_{suffix}"

# Monter le dossier distant
local=True
if local == False:
    if grid: 
        mount_point = os.path.expanduser("~/imagenet_grid")
        try:
            os.makedirs(mount_point, exist_ok=True)
            subprocess.run(["sshfs", "dauce.e@brain-lid-008:Recherche/scripts/S3C/scripts/data/Imagenet_grid_Z", mount_point, "-o", "reconnect"], check=True)
        except:
            pass
    elif orig: 
        mount_point = os.path.expanduser("~/imagenet_orig")
        try:
            os.makedirs(mount_point, exist_ok=True)
            subprocess.run(["sshfs", "dauce.e@brain-lid-008:Recherche/scripts/S3C/scripts/data/Imagenet_Z_orig", mount_point, "-o", "reconnect"], check=True)
        except:
            pass
    else:
        mount_point = os.path.expanduser("~/imagenet")
        try:
            os.makedirs(mount_point, exist_ok=True)
            subprocess.run(["sshfs", "dauce.e@brain-lid-004:Recherche/scripts/S3C/scripts/data/Imagenet_Z", mount_point, "-o", "reconnect"], check=True)
        except:
            pass
    train_dir = os.path.join(mount_point, "train")
    val_dir = os.path.join(mount_point, "val")
else:
    if grid:
        train_dir = "data/Imagenet_grid_Z/train"   # Imagenet Validation set
        val_dir = "data/Imagenet_grid_Z/val"   # Imagenet Validation set
    elif orig:
        train_dir = "data/Imagenet_Z_orig/train"   # Imagenet Validation set
        val_dir = "data/Imagenet_Z_orig/val"   # Imagenet Validation set
    else:
        train_dir = "data/Imagenet_Z/train"   # Imagenet Validation set
        val_dir = "data/Imagenet_Z/val"   # Imagenet Validation set

train_dataset = ImageNetZDataset(train_dir) 
val_dataset   = ImageNetZDataset(val_dir) 

train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
    )
val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
    )

#  SHIFT PREDICTOR

shift_predictor = ShiftPredictor2(emb_dim=embed_dim)
shift_predictor.to(device)
shift_predictor.train()

os.makedirs(save_dir, exist_ok=True)

if train_epochs >= 100:
    weight_decay=3e-4
else:
    weight_decay=1e-3,
optimizer = torch.optim.AdamW(
    shift_predictor.parameters(),
    lr=alpha,
    weight_decay=weight_decay, #0.04,  
)

mse = nn.MSELoss()

schedule = True
if train_epochs > 30:
    n_warm = train_epochs // 6
else:
    n_warm = 5
if schedule:
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warm)
    cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - n_warm)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warm])


log_interval = 100

history = {"epoch": [], "batch": [], "Train loss": [], "Test loss": []}

os.makedirs(save_dir, exist_ok=True)

global_step = 0

for epoch in range(train_epochs):  

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (features, sxs, sys_, labels) in enumerate(pbar):

        batch_size, n = sxs.shape

        #features  : (B, n_saccades_max, 768)
        features = features.to(device)
        labels   = labels.to(device)
        sxs   = sxs.to(device)
        sys_   = sys_.to(device)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, k)
        perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

        idx_t = perms[:, :2]

        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)
        x = sxs[torch.arange(batch_size).unsqueeze(1),  idx_t]                          # (B,)
        y = sys_[torch.arange(batch_size).unsqueeze(1), idx_t]  
        xy = torch.stack([x, y], dim=2)
        shift_target = xy[:,1,:] -  xy[:,0,:]  

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_t = shift_predictor(features_t[:, 0,:], features_t[:, 1,:])                      
            loss = mse(output_t, shift_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            shift_predictor.eval()

            print(f"Epoch {epoch+1:03d} | simple loss = {total_loss / log_interval:.4f}")
            history["epoch"].append(epoch + 1)
            history["batch"].append(batch_idx + 1)
            history["Train loss"].append(total_loss / log_interval)

            total_loss = 0

            total = 0

            running_loss = 0.0

            val_iter = iter(val_loader)                

            with torch.no_grad():
                for n_val in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    #features  : (B, n_saccades_max, 768)
                    features = features.to(device)
                    labels   = labels.to(device)
                    sxs   = sxs.to(device)
                    sys_   = sys_.to(device)

                    # Génère des indices aléatoires pour chaque échantillon du batch
                    # Shape : (batch_size, k)
                    batch_size, n = sxs.shape

                    perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

                    idx_t = perms[:, :2] 

                    x = sxs[torch.arange(batch_size).unsqueeze(1),  idx_t]                          # (B,)
                    y = sys_[torch.arange(batch_size).unsqueeze(1), idx_t]  
                    xy = torch.stack([x, y], dim=2)
                    shift_target = xy[:,1,:] -  xy[:,0,:]  

                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_t = shift_predictor(features_t[:, 0,:], features_t[:, 1,:])                      
                        loss = mse(output_t, shift_target)

                    running_loss += loss.item()

                    total += labels.size(0)


            history["Test loss"].append(running_loss / total)

            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            shift_predictor.train()

    if epoch % 10 == 9:
        torch.save({
                "epoch": epoch,
                "history": history,
                "shift_predictor": shift_predictor.state_dict(),
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

    if schedule:
        scheduler.step()


