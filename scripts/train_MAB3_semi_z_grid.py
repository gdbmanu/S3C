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
from s3c.utils.training import sigreg #SIGReg

import timm

from PIL import Image

from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768

# Monter le dossier distant
local=True
if local == False:
    mount_point = os.path.expanduser("~/imagenet_grid")

    try:
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(["sshfs", "dauce.e@brain-lid-008:Recherche/scripts/S3C/scripts/data/Imagenet_grid_Z", mount_point, "-o", "reconnect"], check=True)
    except:
        pass

    # Ton code ici, en utilisant mount_point
    train_dir = os.path.join(mount_point, "train")
    val_dir = os.path.join(mount_point, "val")
else:
    train_dir = "data/Imagenet_grid_Z/train"   # Imagenet Validation set
    val_dir = "data/Imagenet_grid_Z/val"   # Imagenet Validation set




epoch_teacher = 20

zoom = 1.5

std = 0.5 / zoom 


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

n_sab = 2
k = 3
n_heads = 12

n_saccades_max = 121
n_uplet_student = 3
n_uplet_teacher = 8
n_student_draws = 6
n_teacher_draws = 2

train_epochs = 100
lam=0.5           # λ : trade-off JEPA / SIGReg

save_dir = f"../checkpoints/checkpoints_260427_EMA_Xattn_MAB{k}_semi_z_lam{lam}_sab{n_sab}_grid_LeJ"

mab_transformer = FovealSetTransformer(input_dim=embed_dim, 
                 n_heads=n_heads, n_sab=n_sab, k=k, predict=False)
                 #n_heads=12, n_sab=4, predict=False)
mab_transformer.to(device)
mab_transformer.train()

linear_head = nn.Sequential(
                nn.LayerNorm(k*embed_dim),
                nn.Linear(k*embed_dim, 1000),
            )
linear_head.to(device)
linear_head.train()

os.makedirs(save_dir, exist_ok=True)

# Optimiseur
#optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.001, momentum=0.9)

optimizer = torch.optim.AdamW(
    mab_transformer.parameters(),
    lr=3e-5,              #
    weight_decay=1e-3, #0.04,  
)

linear_optimizer = torch.optim.AdamW(
    linear_head.parameters(),
    lr=1e-4,              #
    weight_decay=1e-3, #0.04,  
)



scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

#sigreg = SIGReg()


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

global_step = 0



for epoch in range(train_epochs):  

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (features, sxs, sys_, labels) in enumerate(pbar):

        #features = features.to(device)          # (B, n_saccades_max, 768)
        labels   = labels.to(device)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, k)
        
        batch_size, n = sxs.shape
        perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])
        idx_s = perms[:, :n_uplet_student * n_student_draws]                                    # (batch_size, n_uplet_student)
        idx_t = perms[:, n_uplet_student * n_student_draws:n_uplet_student*n_student_draws + n_uplet_teacher*n_teacher_draws]   # (batch_size, n_uplet_teacher)


        features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_s = torch.stack([mab_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
            output_t = torch.stack([mab_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)])
            centers = output_t.mean(dim=0)

            loss_jepa = 0
            loss_sigreg = 0
            for i in range(n_student_draws):
                #for j in range(n_teacher_draws):
                #    loss_jepa += mse(output_s[i], output_t[j])
                for j in range(k):
                    loss_jepa += mse(output_s[i,:,j,:], centers[:,j,:]) 
                loss_sigreg += sigreg(output_s[i].view(batch_size, k*embed_dim).float(), global_step)
                #global_step += 1 # !!! TEST 2 !!!
            #for i in range(n_teacher_draws):
            #    loss_sigreg += sigreg(output_t[i].float(), global_step)
            #loss_sigreg += sigreg(centers.view(batch_size, k*embed_dim).float(), global_step)
            global_step += 1

            output_t_head = linear_head(centers.view(batch_size, k*embed_dim).detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
            loss_label = criterion(output_t_head, labels)

            loss = (1 - lam) * loss_jepa + lam * loss_sigreg 

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
                for n_val in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    labels   = labels.to(device)

                    batch_size, n = sxs.shape
                    perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])
                    idx_s = perms[:, :n_uplet_student * n_student_draws]                                    # (batch_size, n_uplet_student)
                    idx_t = perms[:, n_uplet_student * n_student_draws:n_uplet_student*n_student_draws + n_uplet_teacher*n_teacher_draws]   # (batch_size, n_uplet_teacher)


                    # Sélectionne les 3 valeurs pour x, y et z
                    features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_s = torch.stack([mab_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
                        output_t = torch.stack([mab_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)])
                        centers = output_t.mean(dim=0)

                        loss_jepa = 0
                        loss_sigreg = 0
                        for i in range(n_student_draws):
                            #for j in range(n_teacher_draws):
                            #    loss_jepa += mse(output_s[i], output_t[j])
                            for j in range(k):
                                loss_jepa += mse(output_s[i,:,j,:], centers[:,j,:]) 
                            loss_sigreg += sigreg(output_s[i].view(batch_size, k*embed_dim).float(), global_step)
                            # global_step += 1 # !!! TEST 2 !!!
                        #for i in range(n_teacher_draws):
                        #    loss_sigreg += sigreg(output_t[i].float(), global_step)
                        #loss_sigreg += sigreg(centers.view(batch_size, k*embed_dim).float(), global_step)
                        global_step += 1

                        output_t_head = linear_head(centers.view(batch_size, k*embed_dim).detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                        loss_label = criterion(output_t_head, labels)

                        loss = (1 - lam) * loss_jepa + lam * loss_sigreg 

                        if n_val == 0:
                            ratio = lam * loss_sigreg.item() / ((1 - lam) * loss_jepa.item() + 1e-8)
                            print(f"ratio sigreg/jepa = {ratio:.2f}")

                    preds = output_t_head.argmax(dim=1)
                    #print(preds)

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
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            mab_transformer.train()
            linear_head.train()

    if epoch % 10 == 9:
        torch.save({
                "epoch": epoch,
                "history": history,
                "mab_transformer": mab_transformer.state_dict(),
                "linear_head": linear_head.state_dict()
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
        

    if schedule:
        scheduler.step()


