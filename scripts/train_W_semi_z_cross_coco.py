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

from s3c.models.heads import WhatTransformer, AttentionPooling, ShiftPredictor #FovealSetTransformer
from s3c.data.datasets import CocoZDataset
from s3c.models.heads import  PosPredictor

import timm

from PIL import Image

from tqdm import tqdm

from datetime import datetime


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256 #128 #
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768
bottleneck_dim = 768

epoch_teacher = 20

zoom = 1.5
std = 0.5 / zoom 

n_sab = 4 #

n_heads = 12

n_saccades_max = 30 
n_uplet_teacher = 15
n_probes = 3

orig = False
flat = False

train_epochs = 3000
mu = 1               # spatial probe weight

alpha = 1e-3
delta = 3e-4

inv_temp = 1
stop_gradient = False

index_embeddings = True 
synset_level = 4

label_smoothing = 0.1
label_mask = 0.

residual = True
skip_ff_l = True

suffix = ""
suffix = suffix + f"_a{alpha}"
suffix = suffix + f"_d{delta}"


if stop_gradient : suffix = suffix + "_STOP"
if inv_temp != 1: suffix = suffix + f"_IT{inv_temp}"
if bottleneck_dim != 768 : suffix = suffix + f"_BOTTLE{bottleneck_dim}"

if label_smoothing != 0.:
    suffix = suffix + f"_SMOOTH{label_smoothing}"
if index_embeddings:
    suffix = suffix + "_INDEX"

if residual:
    suffix = suffix + f"_RESID"
if skip_ff_l:
    suffix = suffix + "_SKIP"

if orig: suffix = suffix + "_ORIG"
if flat: suffix = suffix + "_FLAT"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_W_sab{n_sab}_{suffix}_t{n_uplet_teacher}_coco"

# Monter le dossier distant

if orig:
    if flat:
        train_dir = "data/COCO_Z_orig_flat/train"   # Imagenet Validation set
        val_dir = "data/COCO_Z_orig_flat/val"   # Imagenet Validation set
    else:
        train_dir = "data/COCO_Z_orig/train"   # Imagenet Validation set
        val_dir = "data/COCO_Z_orig/val"   # Imagenet Validation set
else:
    train_dir = "data/COCO_Z/train"   # Imagenet Validation set
    val_dir = "data/COCO_Z/val"   # Imagenet Validation set

train_dataset = CocoZDataset(train_dir) 
val_dataset   = CocoZDataset(val_dir) 

n_labels = 80

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


emb = None
frozen_emb = False

ist_transformer = WhatTransformer(n_heads=n_heads, n_blocks=n_sab, pretrained_embeddings=emb, residual=residual, skip_ff_l=skip_ff_l,
                                                label_smoothing=label_smoothing, label_mask=label_mask, n_classes=n_labels, frozen_emb=frozen_emb)

ist_transformer.to(device)
ist_transformer.train()

# LINEAR PROBE

linear_head = nn.Sequential(
                nn.LayerNorm(embed_dim),                  
                nn.Linear(embed_dim, n_labels),
            )

linear_head.to(device)
linear_head.train()   

os.makedirs(save_dir, exist_ok=True)


if train_epochs >= 100:
    weight_decay=3e-4
else:
    weight_decay=1e-3

# optimizer = torch.optim.AdamW(
#     ist_transformer.parameters(),
#     lr = delta, #1e-5},
#     weight_decay=0.04,  
# )

decay, no_decay = [], []
for n, p in ist_transformer.named_parameters():
    (no_decay if p.ndim < 2 or "cls_token" in n or "embedding" in n else decay).append(p)
    
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=delta)

linear_optimizer = torch.optim.SGD(
    linear_head.parameters(), 
    lr = alpha,
    momentum = 0.98, 
    weight_decay=weight_decay, 
    nesterov = True  
)

#scaler = torch.cuda.amp.GradScaler()
if label_smoothing:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-1)
else:
    criterion = nn.CrossEntropyLoss()
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

history = {"epoch": [], "batch": [], "loss": [],
        "loss_label": [], "loss_z_pos": [], 
        "cosine_sim": [], 
        "classif": [],}

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

        # images avec au moins une vue labellisée (les autres sont écartées du batch)
        has_lab = labels >= 0
        valid = has_lab.any(dim=1)
        if not valid.any():
            continue
        features, labels, has_lab = features[valid], labels[valid], has_lab[valid]
        sxs, sys_ = sxs[valid], sys_[valid]
        B, n = labels.shape
        ar = torch.arange(B, device=device)

        rnd = torch.rand(B, n, device=device)   # scores aléatoires, utilisés pour les deux tirages

        # 2. cible : une vue labellisée tirée au hasard par image
        j = rnd.masked_fill(~has_lab, 2.0).argmin(dim=1)   # (B,)
        z_star   = features[ar, j]                         # (B, 768)
        lab_star = labels[ar, j]                           # (B,)

        # 1. entrées : b_teacher vues SANS label tirées au hasard
        n_unlab = (~has_lab).sum(dim=1).min().item()
        if n_unlab == 0:
            continue
        b_teacher = min(np.random.randint(1, n_uplet_teacher), n_unlab)
        idx_t = rnd.masked_fill(has_lab, 2.0).argsort(dim=1)[:, :b_teacher]   # (B, b_teacher)
        features_t = features[ar.unsqueeze(1), idx_t]                          # (B, b_teacher, 768)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_t = ist_transformer(features_t, lab_star)

            loss = F.mse_loss(output_t[:,0,:], z_star)
           
            ### LABEL LOSS           
            output_t_head = linear_head(z_star) 
            loss_label = criterion(output_t_head, lab_star)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
        optimizer.step()

        linear_optimizer.zero_grad()
        loss_label.backward()
        linear_optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            ist_transformer.eval()
            linear_head.eval()

            print(f"Epoch {epoch+1:03d} | simple loss = {total_loss / log_interval:.4f}")
            history["epoch"].append(epoch + 1)
            history["batch"].append(batch_idx + 1)
            history["loss"].append(total_loss / log_interval)

            total_loss = 0

            total = 0
            correct = 0.0

            running_label = 0.0
            running_z_pos = 0.0

            running_cosine_sim = 0.0

            val_iter = iter(val_loader)                

            with torch.no_grad():
                for n_val in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    #features  : (B, n_saccades_max, 768)
                    features = features.to(device)
                    labels   = labels.to(device)
                    sxs   = sxs.to(device)
                    sys_   = sys_.to(device)

                    # images avec au moins une vue labellisée (les autres sont écartées du batch)
                    has_lab = labels >= 0
                    valid = has_lab.any(dim=1)
                    if not valid.any():
                        continue
                    features, labels, has_lab = features[valid], labels[valid], has_lab[valid]
                    sxs, sys_ = sxs[valid], sys_[valid]
                    B, n = labels.shape
                    ar = torch.arange(B, device=device)

                    rnd = torch.rand(B, n, device=device)   # scores aléatoires, utilisés pour les deux tirages

                    # 2. cible : une vue labellisée tirée au hasard par image
                    j = rnd.masked_fill(~has_lab, 2.0).argmin(dim=1)   # (B,)
                    z_star   = features[ar, j]                         # (B, 768)
                    lab_star = labels[ar, j]                           # (B,)

                    # 1. entrées : b_teacher vues SANS label tirées au hasard
                    n_unlab = (~has_lab).sum(dim=1).min().item()
                    if n_unlab == 0:
                        continue
                    b_teacher = min(np.random.randint(1, n_uplet_teacher), n_unlab)
                    idx_t = rnd.masked_fill(has_lab, 2.0).argsort(dim=1)[:, :b_teacher]   # (B, b_teacher)
                    features_t = features[ar.unsqueeze(1), idx_t]                          # (B, b_teacher, 768)

                
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_t = ist_transformer(features_t, lab_star)

                        loss = F.mse_loss(output_t[:,0,:], z_star)

                        cosine_sim = F.cosine_similarity(output_t[:,0,:], z_star).mean()

                        ### LABEL LOSS 
                        logits_head = linear_head(output_t[:,0,:]) 

                        loss_label = criterion(logits_head, lab_star)                        
                                                
                                        
                    preds = logits_head.argmax(dim=1)
                    correct += (preds == lab_star).sum().item()

                    running_label += loss_label.item()
                    running_z_pos += loss.item()

                    running_cosine_sim += cosine_sim.item()

                    total += lab_star.size(0)

            print(f"Base accuracy: {100 * correct / total:.2f}%")

            history["classif"].append(100 * correct / total)

            history["loss_label"].append(running_label / total)

            history["loss_z_pos"].append(running_z_pos * batch_size / total)
            history["cosine_sim"].append(running_cosine_sim * batch_size / total)

            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            ist_transformer.train()
            linear_head.train()

    if epoch % 10 == 9:
        torch.save({
                "epoch": epoch,
                "history": history,
                "ist_transformer": ist_transformer.state_dict(),
                "linear_head": linear_head.state_dict(),
            },  os.path.join(save_dir, f"checkpoint.pt"))

    if schedule:
        scheduler.step()


