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
from s3c.data.datasets import ImageNetZDataset
from s3c.models.heads import  PosPredictor
from s3c.utils.training import get_parent_synset

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
grid = False
curriculum = False
finetune = False

if grid:
    n_saccades_max = 121
    n_uplet_teacher = 18

train_epochs = 100
lam = 0.05           # λ : trade-off JEPA / SIGReg
mu = 1               # spatial probe weight

alpha = 3e-7
delta = 3e-6

inv_temp = 1
stop_gradient = False

residual = True
pre_label = False

label_smoothing = 0.5

suffix = ""
suffix = suffix + f"_a{alpha}"
suffix = suffix + f"_d{delta}"

if curriculum or finetune:
    #load_dir = "../checkpoints/260630_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH_APOS2_s3_t5_space"
    assert False # TODO
if curriculum: suffix = suffix + "_CURRI"
if finetune: 
    suffix = suffix + "_FINE"
if grid : suffix = suffix + "_GRID"
    
if stop_gradient : suffix = suffix + "_STOP"
if inv_temp != 1: suffix = suffix + f"_IT{inv_temp}"
if bottleneck_dim != 768 : suffix = suffix + f"_BOTTLE{bottleneck_dim}"

if label_smoothing != 0.:
    suffix = suffix + f"_SMOOTH{label_smoothing}"

if residual:
    suffix = suffix + f"_RESID"
if not pre_label:
    suffix = suffix + "_SKIP"

if orig: suffix = suffix + "_ORIG"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_L_sab{n_sab}_{suffix}_t{n_uplet_teacher}_space"

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

ist_transformer = WhatTransformer(n_heads=n_heads, n_blocks=n_sab, residual=residual,
                                                label_smoothing=label_smoothing)

# LINEAR PROBE

linear_head = nn.Sequential(
                nn.LayerNorm(embed_dim),                  
                nn.Linear(embed_dim, 1000),
            )


if curriculum or finetune:
    epoch_ist = 30
    checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_ist}.pt")  # exemple
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Vérifie les clés disponibles
    print("Clés du checkpoint :", checkpoint.keys())

    if "ist_transformer" not in checkpoint:
        raise KeyError(f"Aucune clé 'ist_transformer' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["ist_transformer"]
    missing, unexpected = ist_transformer.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (ist_transformer).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "linear_head" not in checkpoint:
        raise KeyError(f"Aucune clé 'linear_head' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["linear_head"]
    missing, unexpected = linear_head.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (linear_head).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)


ist_transformer.to(device)
ist_transformer.train()

# LINEAR PROBE

linear_head.to(device)
linear_head.train()   

os.makedirs(save_dir, exist_ok=True)

if finetune:
    optimizer = torch.optim.AdamW([
        {'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
        weight_decay=3e-4, #0.04,  
    )            
    ist_transformer.requires_grad_(False)
else:
    if train_epochs >= 100:
        weight_decay=3e-4
    else:
        weight_decay=1e-3,
    optimizer = torch.optim.AdamW(
        [{'params': ist_transformer.parameters(), 'lr': delta},
         {'params': linear_head.parameters(), 'lr': alpha}], #1e-5},
        weight_decay=weight_decay, #0.04,  
    )

#scaler = torch.cuda.amp.GradScaler()
if label_smoothing:
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
else:
    criterion = nn.CrossEntropyLoss()

schedule = True
if train_epochs > 30:
    n_warm = train_epochs // 6
else:
    n_warm = 5
if schedule:
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warm)
    cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - n_warm)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warm])
    


# %%

log_interval = 100

history = {"epoch": [], "batch": [], "loss": [],
        "loss_label": [], "classif": []}

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

        b_teacher = np.random.randint(1, n_uplet_teacher) # !!
        idx_t = perms[:, :b_teacher]
        

        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_t = ist_transformer(features_t[:, :n_uplet_teacher,:], None)                      
            ### LABEL LOSS           
            output_t_head = linear_head(output_t[:,0,:]) 
            loss_label = criterion(output_t_head, labels)

        optimizer.zero_grad()
        loss_label.backward()
        torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_label.item()
    
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

                    b_teacher = n_uplet_teacher
                    idx_t = perms[:, :b_teacher] 

                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_t = ist_transformer(features_t[:, :n_uplet_teacher,:], None) 
                        
                        ### LABEL LOSS 
                        logits_head = linear_head(output_t[:,0,:]) 

                        loss_label = criterion(logits_head, labels)
                    
                    preds = logits_head.argmax(dim=1)
                    correct += (preds == labels).sum().item()

                    running_label += loss_label.item()

                    total += labels.size(0)

            print(f"Base accuracy: {100 * correct / total:.2f}%")

            history["classif"].append(100 * correct / total)

            history["loss_label"].append(running_label / total)

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
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

    if schedule:
        scheduler.step()


