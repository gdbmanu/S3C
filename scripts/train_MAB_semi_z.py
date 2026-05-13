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


from s3c.models.heads import FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.utils.training import SIGReg

import timm

from PIL import Image

from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768

train_dir = "data/Imagenet_Z/train"   # Imagenet Validation set
val_dir = "data/Imagenet_Z/val"   # Imagenet Validation set

save_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_MAB_semi_z"

epoch_teacher = 20

n_saccades_max = 30

zoom = 1.5

std = 0.5 / zoom 

n_uplet_student = 3
n_uplet_teacher = 8


train_epochs = 100


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

mab_transformer = FovealSetTransformer(input_dim=embed_dim, 
                 n_heads=4, n_sab=2, predict=False)
mab_transformer.to(device)
mab_transformer.train()

linear_head = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 1000),
            )
linear_head.to(device)
linear_head.train()

os.makedirs(save_dir, exist_ok=True)

# Optimiseur
#optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.001, momentum=0.9)

optimizer = torch.optim.AdamW(
    mab_transformer.parameters(),
    lr=3e-4,              #
    weight_decay=1e-4, #0.04,  
)

linear_optimizer = torch.optim.AdamW(
    linear_head.parameters(),
    lr=1e-4,              #
    weight_decay=1e-4, #0.04,  
)



scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

lam=1           # λ : trade-off JEPA / SIGReg
sigreg = SIGReg()


schedule = True
if schedule:
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])


# %%

log_interval = 100

history = {"epoch": [], "batch": [], "loss": [],
        "loss_label": [], "loss_jepa" : [], "loss_sigreg" : [], "classif": []} #, "train_loss_3": []}

os.makedirs(save_dir, exist_ok=True)

for epoch in range(train_epochs):  

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (features, sxs, sys_, labels) in enumerate(pbar):

        #features = features.to(device)          # (B, n_saccades_max, 768)
        labels   = labels.to(device)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, k)
        batch_size, n = sxs.shape
        idx_s = torch.stack([torch.randperm(n_saccades_max)[:3] for _ in range(batch_size)])
        idx_t = torch.stack([torch.randperm(n_saccades_max)[:8] for _ in range(batch_size)])

        # Sélectionne les 3 valeurs pour x, y et z
        features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_s = mab_transformer(features_s)
            output_t = mab_transformer(features_t)
            output_t_head = linear_head(output_t.detach())

            loss_jepa = mse(output_s, output_t)
            loss_sigreg = sigreg(output_s.float()) + sigreg(output_t.float())
            loss_label = criterion(output_t_head, labels)

            loss = loss_jepa + lam * loss_sigreg 


        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(mab_transformer.parameters(), 1.0)
        optimizer.step()

        linear_optimizer.zero_grad()
        loss_label.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(linear_head.parameters(), 1.0)
        linear_optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            mab_transformer.eval()
            linear_head.eval()

            print(f"Epoch {epoch+1:03d} | simple loss = {total_loss / log_interval:.4f}")

            history["epoch"].append(epoch + 1)
            history["batch"].append(batch_idx + 1)
            history["loss"].append(total_loss / log_interval)

            total_loss = 0

            total = 0
            correct = 0.0
            running_sigreg= 0.0
            running_jepa= 0.0
            running_label = 0.0
            val_iter = iter(val_loader)

            with torch.no_grad():
                for _ in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    labels   = labels.to(device)

                    batch_size, n = sxs.shape
                    idx_s = torch.stack([torch.randperm(n_saccades_max)[:3] for _ in range(batch_size)])
                    idx_t = torch.stack([torch.randperm(n_saccades_max)[:8] for _ in range(batch_size)])

                    # Sélectionne les 3 valeurs pour x, y et z
                    features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output_s = mab_transformer(features_s)
                        output_t = mab_transformer(features_t)
                        output_t_head = linear_head(output_t)

                        loss_jepa = mse(output_s, output_t.detach())
                        loss_sigreg = sigreg(output_s.float()) + sigreg(output_t.float())
                        loss_label = criterion(output_t_head.detach(), labels)

                        loss = loss_jepa + lam * loss_sigreg + loss_label

                    preds = output_t_head.argmax(dim=1)
                    print(preds)

                    correct += (preds == labels).sum().item()
                    running_sigreg += loss_sigreg.item()
                    running_jepa += loss_jepa.item()
                    running_label = loss_label.item()

                    total += labels.size(0)

            print(f"Top-1 accuracy: {100 * correct / total:.2f}%")

            history["classif"].append(100 * correct / total)
            history["loss_sigreg"].append(running_sigreg / total)
            history["loss_jepa"].append(running_jepa / total)
            history["loss_label"].append(running_label / total)

            mab_transformer.train()
            linear_head.train()

    if epoch % 9 == 0:
        torch.save({
                "epoch": epoch,
                "history": history,
                "mab_transformer": mab_transformer.state_dict(),
                "linear_head": linear_head.state_dict()
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

    if schedule:
        scheduler.step()


