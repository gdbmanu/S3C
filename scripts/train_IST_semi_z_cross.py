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


from s3c.models.heads import IterativeSeedTransformer #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.utils.training import sigreg #SIGReg

import timm

from PIL import Image

from tqdm import tqdm

from datetime import datetime


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768

epoch_teacher = 20

zoom = 1.5
std = 0.5 / zoom 

n_sab = 2
k = 1
n_heads = 12

n_saccades_max = 30 
n_uplet_student = 3
n_uplet_teacher = 9 #5
n_student_draws = 6
n_teacher_draws = 2

train_epochs = 100
lam = 0.05           # λ : trade-off JEPA / SIGReg

inv_temp = 1
supervised = False
test = False # seed diversity
test3 = False # no sample diversity
strict_global_step = False
wide_views = False
central_integration = False
stop_gradient = False
cross_integration = True
grid = True
curriculum = False

suffix = ""
if supervised : suffix = suffix + "_SUP"
if test : suffix = suffix + "_TEST"
if test3 : suffix = suffix + "_TEST3"
if strict_global_step : suffix = suffix + "_STRICT"
if central_integration : suffix = suffix + "_CENTRAL"
if stop_gradient : suffix = suffix + "_STOP"
if cross_integration : suffix = suffix + "_CROSS"
if wide_views : suffix = suffix + "_WIDE"
if curriculum:
    load_dir = f"../checkpoints/260528_IST1+ABMIL_semi_z_lam0.05_sab2_LeJ{suffix}_s{n_uplet_student}_t{n_uplet_teacher}_(**)"
    suffix = suffix + "_CURRI"
if grid : 
    suffix = suffix + "_GRID"
    n_saccades_max = 121

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_IST{k}+ABMIL_semi_z_lam{lam}_sab{n_sab}_LeJ{suffix}_s{n_uplet_student}_t{n_uplet_teacher}"

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



ist_transformer = IterativeSeedTransformer(input_dim=embed_dim, d_model=embed_dim,
                 n_heads=n_heads, n_seeds=k, n_blocks=n_sab)

attention = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )     

if k>1:
    attentions = nn.ModuleList([
        nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        for _ in range(k)
    ])      

linear_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1000),
            )


if curriculum:
    epoch_ist = 100
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

    if "attention" not in checkpoint:
        raise KeyError(f"Aucune clé 'attention' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["attention"]
    missing, unexpected = attention.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (attention).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "linear_head" not in checkpoint:
        raise KeyError(f"Aucune clé 'linear_head' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["linear_head"]
    missing, unexpected = linear_head.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (linear_head).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if k>1:
        if "attentions" not in checkpoint:
            raise KeyError(f"Aucune clé 'attentions' trouvée dans {checkpoint_path}")
        state_dict = checkpoint["attentions"]
        missing, unexpected = attentions.load_state_dict(state_dict, strict=False)
        print("➡️ Poids chargés (attentions).")
        print("❗ Paramètres manquants :", missing)
        print("⚠️ Paramètres inattendus :", unexpected)



ist_transformer.to(device)
ist_transformer.train()

attention.to(device)
attention.train()

if k>1:
    attentions.to(device)
    attentions.train()

# LINEAR PROBES

linear_head.to(device)
linear_head.train()     

heads_per_seed = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1000) 
            ) for _ in range(k)
        ])

heads_per_seed.to(device)
heads_per_seed.train()

os.makedirs(save_dir, exist_ok=True)

# Optimiseur
#optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.001, momentum=0.9)

'''optimizer = torch.optim.AdamW(
    ist_transformer.parameters(),
    lr=3e-5,              #
    weight_decay=1e-3, #0.04,  
)'''


if supervised:
    if k>1:
        linear_optimizer = torch.optim.AdamW(
            [{'params': ist_transformer.parameters(), 'lr': 3e-5}, #3e-6},
            {'params': attentions.parameters(),       'lr': 1e-4}, #1e-5},
            {'params': attention.parameters(),       'lr': 3e-4}, #1e-5},
            {'params': linear_head.parameters(), 'lr': 3e-4}], #3e-6}], #1e-4}],
            weight_decay=1e-3, #0.04,  
        )
    else:
        linear_optimizer = torch.optim.AdamW(
            [{'params': ist_transformer.parameters(), 'lr': 3e-5}, #3e-6},
            {'params': attention.parameters(),       'lr': 3e-4}, #1e-5},
            {'params': linear_head.parameters(), 'lr': 3e-4}], #3e-6}], #1e-4}],
            weight_decay=1e-3, #0.04,  
        )
else:
    if k>1:
        optimizer = torch.optim.AdamW([
            {'params': ist_transformer.parameters(), 'lr': 3e-5},
            {'params': attentions.parameters(),       'lr': 1e-4}, #1e-5},
            {'params': attention.parameters(),       'lr': 3e-4},
        ], weight_decay=1e-3)
    else:
        optimizer = torch.optim.AdamW([
            {'params': ist_transformer.parameters(), 'lr': 3e-5},
            {'params': attention.parameters(),       'lr': 1e-4},
        ], weight_decay=1e-3)

    linear_optimizer = torch.optim.AdamW(
        linear_head.parameters(),
        lr=1e-4,              #
        weight_decay=1e-3, #0.04,  
    )

seeds_optimizer = torch.optim.AdamW(
    heads_per_seed.parameters(),
    lr=1e-4,              #
    weight_decay=1e-3, #0.04,  
)

#scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

#sigreg = SIGReg()

schedule = True
if schedule:
    if supervised:
        warmup = LinearLR(linear_optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(linear_optimizer, T_max=train_epochs - 5)
        scheduler = SequentialLR(linear_optimizer, schedulers=[warmup, cosine], milestones=[5])
    else:
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
        #wide = torch.tensor([0, 10, 60, 110, 120] * n_teacher_draws).repeat(batch_size, 1) 
        regular_grid = [12, 20, 60, 100, 108, 38, 58, 62, 82]
        wide = torch.tensor(regular_grid[:n_uplet_teacher] * n_teacher_draws).repeat(batch_size, 1) 
        idx_s = perms[:, :n_uplet_student * n_student_draws]       
        if wide_views:
            idx_t = wide[:, : n_uplet_teacher * n_teacher_draws]   # (batch_size, n_uplet_teacher)
        else:
            idx_t = perms[:, n_uplet_student * n_student_draws:n_uplet_student*n_student_draws + n_uplet_teacher*n_teacher_draws]   # (batch_size, n_uplet_teacher)

        features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
            output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)])
            centers = output_t.mean(dim=0)

            loss_jepa = 0
            loss_sigreg = 0
            
            if test: 
                if k>1:
                    for j in range(k): # seeds loop
                        for i in range(n_student_draws):           
                            loss_sigreg += sigreg(output_s[i,:,j,:].float(), global_step) # !! TEST diversité sur les seeds
                        if not strict_global_step:
                            global_step += 1 # !!! TEST !!!
                else:
                    assert False # not consistent

            if k > 1:
                w_c_ref = attention(centers)                      # (B, k, 1)
                w = torch.softmax(w_c_ref * inv_temp, dim=1)                 # (B, k, 1)
                z_centers = (w * centers).sum(dim=1)           # (B, d_model)
            else:
                z_centers = centers.squeeze(dim=1)

            z_draws = []
            for i in range(n_student_draws):
                if k > 1:
                    w_ref = attention(output_s[i]) 
                    w = torch.softmax(w_ref, dim=1) 
                    z_draw = (w * output_s[i]).sum(dim=1)
                else:
                    z_draw = output_s[i].squeeze(dim=1)
                if not cross_integration:
                    if stop_gradient:
                        loss_jepa += mse(z_draw, z_centers.detach())
                    else:
                        loss_jepa += mse(z_draw, z_centers)
                else:
                    z_draws.append(output_s[i].clone())
                if not test3:
                    loss_sigreg += sigreg(z_draw.float(), global_step) ## !! TEST diversité sur les draws
                    if central_integration:
                        loss_sigreg += sigreg(z_centers.float(), global_step)
                    if not strict_global_step:
                        global_step += 1

            if cross_integration and k==1:
                z_stacked = torch.stack(z_draws, dim=1)
                w_ref = attention(z_stacked) 
                w = torch.softmax(w_ref, dim=1) 
                global_z_draw = (w * z_stacked).sum(dim=1)
                if stop_gradient:
                    loss_jepa = mse(global_z_draw.squeeze(dim=1), z_centers.detach())
                else:
                    loss_jepa = mse(global_z_draw.squeeze(dim=1), z_centers)
            else:
                z_seeds = []
                z_stacked = torch.stack(z_draws, dim=1)

                for seed_idx in range(k):
                    # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                    views_seed = z_stacked[:, :, seed_idx, :]

                    # ABMIL spécifique à ce seed
                    w_ref = attentions[seed_idx](views_seed)        # (B, n_draws, 1)
                    w     = torch.softmax(w_ref, dim=1)             # softmax sur n_draws
                    z     = (w * views_seed).sum(dim=1)             # (B, d)
                    z_seeds.append(z)
             
                z_cross = torch.stack(z_seeds, dim=1)
                w_ref = attention(z_cross) 
                w = torch.softmax(w_ref, dim=1) 
                global_z_draw = (w * z_cross).sum(dim=1)
                if stop_gradient:
                    loss_jepa = mse(global_z_draw, z_centers.detach())
                else:
                    loss_jepa = mse(global_z_draw, z_centers)

            if strict_global_step:
                global_step += 1

            #loss_sigreg += sigreg(centers.view(batch_size, k*embed_dim).float(), global_step)
            #global_step += 1

            loss_seeds = 0
            for j in range(k):
                output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                loss_seeds += criterion(output_t_seed, labels)

            if supervised:
                output_t_head = linear_head(z_centers)
                loss_label = loss = (1 - lam) * loss_jepa + lam * loss_sigreg + criterion(output_t_head, labels)
            else:
                output_t_head = linear_head(z_centers.detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                loss_label = criterion(output_t_head, labels)
                loss = (1 - lam) * loss_jepa + lam * loss_sigreg 

        if not supervised:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
            optimizer.step()

        linear_optimizer.zero_grad()
        loss_label.backward()
        #grad_norm = torch.nn.utils.clip_grad_norm_(linear_head.parameters(), 1.0)
        linear_optimizer.step()

        seeds_optimizer.zero_grad()
        loss_seeds.backward()
        #grad_norm = torch.nn.utils.clip_grad_norm_(seeds_optimizer.parameters(), 1.0)
        seeds_optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            ist_transformer.eval()
            linear_head.eval()
            heads_per_seed.eval()
            attention.eval()
            if k>1:
                attentions.eval()

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
            seeds_correct = [0.0 for j in range(k)]
            val_iter = iter(val_loader)

            with torch.no_grad():
                for n_val in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    labels   = labels.to(device)

                    batch_size, n = sxs.shape
                    perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])
                    regular_grid = [12, 20, 60, 100, 108, 38, 58, 62, 82]
                    wide = torch.tensor(regular_grid[:n_uplet_teacher] * n_teacher_draws).repeat(batch_size, 1)                     
                    idx_s = perms[:, :n_uplet_student * n_student_draws]                                    # (batch_size, n_uplet_student)
                    if wide_views:
                        idx_t = wide[:, : n_uplet_teacher * n_teacher_draws]   # (batch_size, n_uplet_teacher)
                    else:
                        idx_t = perms[:, n_uplet_student * n_student_draws:n_uplet_student*n_student_draws + n_uplet_teacher*n_teacher_draws]   # (batch_size, n_uplet_teacher)

                    # Sélectionne les 3 valeurs pour x, y et z
                    features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
                        output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)])
                        centers = output_t.mean(dim=0)

                    loss_jepa = torch.tensor(0.).to(device)
                    loss_sigreg = torch.tensor(0.).to(device)

                    if test: 
                        if k>1:
                            for j in range(k): # seeds loop
                                for i in range(n_student_draws):           
                                    loss_sigreg += sigreg(output_s[i,:,j,:].float(), global_step) # !! TEST diversité sur les seeds
                                if not strict_global_step:
                                    global_step += 1 # !!! TEST !!!
                        else:
                            assert False # not consistent

                    if k > 1:
                        w_c_ref = attention(centers)                      # (B, k, 1)
                        w_c = torch.softmax(w_c_ref * inv_temp, dim=1)      # (B, k, 1)
                        z_centers = (w_c * centers).sum(dim=1)           # (B, d_model)
                    else:
                        z_centers = centers.squeeze(dim=1)

                    z_draws = []
                    for i in range(n_student_draws):
                        if k > 1:
                            w_ref = attention(output_s[i]) 
                            w = torch.softmax(w_ref, dim=1) 
                            z_draw = (w * output_s[i]).sum(dim=1)
                        else:
                            z_draw = output_s[i].squeeze(dim=1)
                        if not cross_integration:
                            if stop_gradient:
                                loss_jepa += mse(z_draw, z_centers.detach())
                            else:
                                loss_jepa += mse(z_draw, z_centers)
                        else:
                            z_draws.append(output_s[i].clone())
                        if not test3:
                            loss_sigreg += sigreg(z_draw.float(), global_step) ## !! TEST diversité sur les draws
                            if central_integration:
                                loss_sigreg += sigreg(z_centers.float(), global_step)
                            if not strict_global_step:
                                global_step += 1

                        if not strict_global_step:
                                    global_step += 1

                    if cross_integration and k==1:
                        z_stacked = torch.stack(z_draws, dim=1)
                        w_ref = attention(z_stacked) 
                        w = torch.softmax(w_ref, dim=1) 
                        global_z_draw = (w * z_stacked).sum(dim=1)
                        loss_jepa = mse(global_z_draw.squeeze(dim=1), z_centers)
                    else:
                        z_seeds = []
                        z_stacked = torch.stack(z_draws, dim=1)

                        mem_w = []
                        for seed_idx in range(k):
                            # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                            views_seed = z_stacked[:, :, seed_idx, :]

                            # ABMIL spécifique à ce seed
                            w_ref = attentions[seed_idx](views_seed)        # (B, n_draws, 1)
                            w     = torch.softmax(w_ref, dim=1)             # softmax sur n_draws
                            mem_w.append(w)
                            z     = (w * views_seed).sum(dim=1)             # (B, d)
                            z_seeds.append(z)

                    
                        z_cross = torch.stack(z_seeds, dim=1)
                        w_ref = attention(z_cross) 
                        w = torch.softmax(w_ref, dim=1) 
                        global_z_draw = (w * z_cross).sum(dim=1)
                        loss_jepa = mse(global_z_draw, z_centers)
                        

                            
                    if strict_global_step:
                        global_step += 1

                    #loss_sigreg += sigreg(centers.view(batch_size, k*embed_dim).float(), global_step)
                    #global_step += 1

                    loss_seeds = 0
                    for j in range(k):
                        output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                        preds = output_t_seed.argmax(dim=1)
                        seeds_correct[j] += (preds == labels).sum().item()

                    output_t_head = linear_head(z_centers.detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                    loss_label = criterion(output_t_head, labels)

                    loss = (1 - lam) * loss_jepa + lam * loss_sigreg 

                    if n_val == 0:
                        ratio = lam * loss_sigreg.item() / ((1 - lam) * loss_jepa.item() + 1e-8)
                        print(f"ratio sigreg/jepa = {ratio:.2f}")
                        if k>1 and not cross_integration:
                            print(w_c[0,...].detach().float().cpu().numpy())
                        elif cross_integration:
                            if k>1:
                                for i in range(k):
                                    print(f"seed {i}", mem_w[i][0,...].detach().float().cpu().numpy())
                                print("global", w[0,...].detach().float().cpu().numpy())
                            else:
                                print(w[0,...].detach().float().cpu().numpy())
                    
                    preds = output_t_head.argmax(dim=1)
                    #print(preds)

                    correct += (preds == labels).sum().item()
                    running_sigreg += loss_sigreg.item()
                    running_jepa += loss_jepa.item()
                    running_label = loss_label.item()

                    total += labels.size(0)

            print(f"Top-1 accuracy: {100 * correct / total:.2f}%")
            for j in range(k):
                print(f"Seed {j} accuracy : {100 * seeds_correct[j] / total:.2f}%")

            history["classif"].append(100 * correct / total)
            history["loss_sigreg"].append(running_sigreg / total)
            history["loss_jepa"].append(running_jepa / total)
            history["loss_label"].append(running_label / total)
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            ist_transformer.train()
            linear_head.train()
            heads_per_seed.train()
            attention.train()
            if k>1:
                attentions.train()

    if epoch % 10 == 9:
        if k>1:
            torch.save({
                    "epoch": epoch,
                    "history": history,
                    "ist_transformer": ist_transformer.state_dict(),
                    "attention": attention.state_dict(),
                    "attentions": attentions.state_dict(),
                    "linear_head": linear_head.state_dict()
                },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
        else:
            torch.save({
                    "epoch": epoch,
                    "history": history,
                    "ist_transformer": ist_transformer.state_dict(),
                    "attention": attention.state_dict(),
                    "linear_head": linear_head.state_dict()
                },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))     

    if schedule:
        scheduler.step()


