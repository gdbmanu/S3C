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


from s3c.models.heads import IterativeSeedTransformer, AttentionPooling #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.utils.training import sigreg, vicReg_seed #SIGReg
from s3c.models.heads import PosPredictor

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
bottleneck_dim = 768

epoch_teacher = 20

zoom = 1.5
std = 0.5 / zoom 

n_sab = 2
self_att = False

k = 3       # n_seeds
n_heads = 12

n_saccades_max = 30 
n_uplet_student = 3
n_uplet_teacher = 5
n_student_draws = 4
n_teacher_draws = 3

orig = False

train_epochs = 30
lam = 0.05           # λ : trade-off JEPA / SIGReg
mu = 1               # spatial probe weight

inv_temp = 1
stop_gradient = False

supervised = False
test = False # seed diversity
center_test = False # center seed consistency
vicreg = False # more seed diversity
test3 = False # no sample diversity
strict_global_step = False
cross_integration = True # cross_draws_integration

grid = False
curriculum = False

suffix = ""
if supervised : suffix = suffix + "_SUP"

if self_att : suffix = suffix + '_SELF'

if test : suffix = suffix + "_TEST"
if center_test : suffix = suffix + "_CENTER"

if vicreg: suffix = suffix + "_VICREG"
if test3 : suffix = suffix + "_TEST3"
if strict_global_step : suffix = suffix + "_STRICT"
if cross_integration : suffix = suffix + "_CROSS"
if curriculum:
    load_dir = f"../checkpoints/260528_IST1+ABMIL_semi_z_lam0.05_sab2_LeJ{suffix}_s{n_uplet_student}_t{n_uplet_teacher}_(**)"
    suffix = suffix + "_CURRI"
if grid : 
    suffix = suffix + "_GRID"
    n_saccades_max = 121
if stop_gradient : suffix = suffix + "_STOP"
if inv_temp != 1: suffix = suffix + f"_IT{inv_temp}"
if bottleneck_dim != 768 : suffix = suffix + f"_BOTTLE{bottleneck_dim}"

if orig: suffix = suffix + "_ORIG"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_IST{k}+ABMIL_semi_z_lam{lam}_mu_{mu}_sab{n_sab}_LeJ{suffix}_s{n_uplet_student}_t{n_uplet_teacher}_space"

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

ist_transformer = IterativeSeedTransformer(input_dim=embed_dim, d_model=embed_dim,
                 n_heads=n_heads, n_seeds=k, n_blocks=n_sab, self_att=self_att)


draws_attention = AttentionPooling(embed_dim, inv_temp=inv_temp)

pos_predictor = PosPredictor(embed_dim, k)

# LINEAR PROBE
linear_head = nn.Sequential(
                nn.Unflatten(1, (k, embed_dim)),          # (B, k*d) → (B, k, d)
                nn.LayerNorm(embed_dim),                  # norm par seed ✓
                nn.Flatten(1),    
                nn.Linear(k * embed_dim, 1000),
            )

seeds_mlp = nn.Sequential(
    nn.Unflatten(1, (k, embed_dim)),          # (B, k*d) → (B, k, d)
    nn.LayerNorm(embed_dim),                  # norm par seed ✓
    nn.Flatten(1),                            # (B, k, d) → (B, k*d)
    nn.Linear(k * embed_dim, k * embed_dim),
    nn.ReLU(),
    nn.Linear(k * embed_dim, k * embed_dim),
    nn.ReLU(),
    nn.Linear(k * embed_dim, bottleneck_dim),
)

if k>1:                                     # cross-draws integration (seed diversity)
    # LINEAR PROBES
    heads_per_seed = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1000) 
            ) for _ in range(k)
        ])

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

    if "draws_attention" not in checkpoint:
        raise KeyError(f"Aucune clé 'draws_attention' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["draws_attention"]
    missing, unexpected = draws_attention.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (draws_attention).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "linear_head" not in checkpoint:
        raise KeyError(f"Aucune clé 'linear_head' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["linear_head"]
    missing, unexpected = linear_head.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (linear_head).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "pos_predictor" not in checkpoint:
        raise KeyError(f"Aucune clé 'pos_predictor' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["pos_predictor"]
    missing, unexpected = pos_predictor.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (pos_predictor).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "seeds_mlp" not in checkpoint:
        raise KeyError(f"Aucune clé 'seeds_mlp' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["seeds_mlp"]
    missing, unexpected = seeds_mlp.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (seeds_mlp).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)




ist_transformer.to(device)
ist_transformer.train()

draws_attention.to(device)
draws_attention.train()

# LINEAR PROBE

linear_head.to(device)
linear_head.train()   

pos_predictor.to(device)
pos_predictor.train()

seeds_mlp.to(device)
seeds_mlp.train()

if k>1:
    heads_per_seed.to(device)
    heads_per_seed.train()

os.makedirs(save_dir, exist_ok=True)

if supervised:
    linear_optimizer = torch.optim.AdamW(
        [{'params': ist_transformer.parameters(), 'lr': 3e-5}, #3e-6},
        {'params': draws_attention.parameters(),       'lr': 1e-4}, #1e-5},
        {'params': pos_predictor.parameters(),       'lr': 1e-4}, #1e-5},
        {'params': seeds_mlp.parameters(),       'lr': 3e-4}, #1e-5},
        {'params': linear_head.parameters(), 'lr': 1e-4}], #3e-6}], #1e-4}],
        weight_decay=1e-3, #0.04,  
    )
else:
    optimizer = torch.optim.AdamW([
        {'params': ist_transformer.parameters(), 'lr': 3e-5},
        {'params': draws_attention.parameters(),       'lr': 1e-4}, #1e-5},
        {'params': pos_predictor.parameters(),       'lr': 1e-4},
        {'params': seeds_mlp.parameters(),       'lr': 3e-4},
    ], weight_decay=1e-3)

    linear_optimizer = torch.optim.AdamW(
        linear_head.parameters(),
        lr=1e-4,              #
        weight_decay=1e-3, #0.04,  
    )

if k > 1:
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
        "loss_label": [], "loss_jepa" : [], "loss_sigreg" : [], "loss_pos": []}
for j in range(k):
    history[f"classif {j}"] = []
history[f"classif"] = []

os.makedirs(save_dir, exist_ok=True)

global_step = 0

for epoch in range(train_epochs):  

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (features, sxs, sys_, labels) in enumerate(pbar):

        #features  : (B, n_saccades_max, 768)
        labels   = labels.to(device)
        sxs   = sxs.to(device)
        sys_   = sys_.to(device)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, k)
        batch_size, n = sxs.shape
        perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

        b_student = n_uplet_student * n_student_draws
        idx_s = perms[:, :b_student]    
        b_teacher = b_student + n_uplet_teacher*n_teacher_draws
        idx_t = perms[:, b_student:b_teacher] 
        idx_probe = perms[:, b_teacher] 
        assert b_teacher + 1 <= n_saccades_max

        features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        x_probe = sxs[torch.arange(batch_size), idx_probe].to(device)
        y_probe = sys_[torch.arange(batch_size), idx_probe].to(device)
        z_probe = features[torch.arange(batch_size), idx_probe, :].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if n_student_draws > 0:
                output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
            output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)], dim=1)
            #
            if cross_integration:
                if k == 1:
                    centers, _ = draws_attention(output_t.squeeze(dim=2)) 
                else:
                    center_seeds = []
                    for seed_idx in range(k):
                        # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                        z, _ = draws_attention(output_t[:, :, seed_idx, :])           # (B, d)
                        center_seeds.append(z)
                    centers = torch.stack(center_seeds, dim=1)
            else:
                centers = output_t.mean(dim=1)

            pos_pred = pos_predictor(centers, z_probe)

            pos_target = torch.stack([x_probe, y_probe], dim=1)   # (B, 2)
            loss_pos = F.mse_loss(pos_pred, pos_target)

            z_center = seeds_mlp(centers.view(batch_size, k*embed_dim))

            loss_jepa = torch.tensor(0.).to(device)
            loss_sigreg = torch.tensor(0.).to(device)

            if test and n_student_draws > 0:
                if k>1:
                    for j in range(k): # seeds loop
                        for i in range(n_student_draws):           
                            loss_sigreg += sigreg(output_s[i,:,j,:].float(), global_step) # !! TEST diversité sur les seeds
                        if not strict_global_step:
                            global_step += 1 # !!! TEST !!!
                else:
                    assert False # not consistent

            if n_student_draws > 0:
                z_draws = []
                for i in range(n_student_draws):
                    z_draw = seeds_mlp(output_s[i].view(batch_size, k*embed_dim))
                    if stop_gradient:
                        loss_jepa += mse(z_draw, z_center.detach())
                    else:
                        loss_jepa += mse(z_draw, z_center) 
                    
                    if not test3:
                        loss_sigreg += sigreg(z_draw.float(), global_step) ## !! TEST diversité sur les draws
                        if not strict_global_step:
                            global_step += 1

            if strict_global_step:
                global_step += 1


            if k>1:
                loss_seeds = 0
                for j in range(k):
                    output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                    loss_seeds += criterion(output_t_seed, labels)

            if supervised:
                #output_t_head = linear_head(z_center)
                output_t_head = linear_head(centers.view(batch_size, k * embed_dim))
                #loss_label = loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos + criterion(output_t_head, labels)
                loss_label = loss = criterion(output_t_head, labels)
            else:
                #output_t_head = linear_head(z_center.detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                output_t_head = linear_head(centers.view(batch_size, k * embed_dim).detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                loss_label = criterion(output_t_head, labels)
                loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos

        if not supervised:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
            if k>1:
                torch.nn.utils.clip_grad_norm_(draws_attention.parameters(), 1.0)
            optimizer.step()

        linear_optimizer.zero_grad()
        loss_label.backward()
        #grad_norm = torch.nn.utils.clip_grad_norm_(linear_head.parameters(), 1.0)
        linear_optimizer.step()

        if k>1:
            seeds_optimizer.zero_grad()
            loss_seeds.backward()
            #grad_norm = torch.nn.utils.clip_grad_norm_(seeds_optimizer.parameters(), 1.0)
            seeds_optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            ist_transformer.eval()
            linear_head.eval()
            draws_attention.eval()
            seeds_mlp.eval()                
            if k>1:
                heads_per_seed.eval()

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
            running_pos = 0.0
            seeds_correct = [0.0 for j in range(k)]
            val_iter = iter(val_loader)

            with torch.no_grad():
                for n_val in range(5):
                    features, sxs, sys_, labels = next(val_iter)
                    #features  : (B, n_saccades_max, 768)
                    labels   = labels.to(device)
                    sxs   = sxs.to(device)
                    sys_   = sys_.to(device)

                    # Génère des indices aléatoires pour chaque échantillon du batch
                    # Shape : (batch_size, k)
                    batch_size, n = sxs.shape
                    perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

                    b_student = n_uplet_student * n_student_draws
                    idx_s = perms[:, :b_student]    
                    b_teacher = b_student + n_uplet_teacher * n_teacher_draws
                    idx_t = perms[:, b_student:b_teacher] 
                    idx_probe = perms[:, b_teacher] 
                    assert b_teacher + 1 <= n_saccades_max

                    features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    x_probe = sxs[torch.arange(batch_size), idx_probe].to(device)
                    y_probe = sys_[torch.arange(batch_size), idx_probe].to(device)
                    z_probe = features[torch.arange(batch_size), idx_probe, :].to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if n_student_draws > 0:
                            output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:]) for i in range(n_student_draws)])
                        output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:]) for i in range(n_teacher_draws)], dim=1)
                        #
                        mem_w = []
                        if cross_integration:
                            if k == 1:
                                centers, w = draws_attention(output_t.squeeze(dim=2)) 
                                mem_w = w
                            else:
                                center_seeds = []
                                for seed_idx in range(k):
                                    # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                                    z, w = draws_attention(output_t[:, :, seed_idx, :])           # (B, d)
                                    mem_w += [w]
                                    center_seeds.append(z)
                                centers = torch.stack(center_seeds, dim=1)
                        else:
                            centers = output_t.mean(dim=1)     

                        pos_pred = pos_predictor(centers, z_probe)

                        pos_target = torch.stack([x_probe, y_probe], dim=1)   # (B, 2)
                        loss_pos = F.mse_loss(pos_pred, pos_target)

                        z_center = seeds_mlp(centers.view(batch_size, k*embed_dim))

                        loss_jepa = torch.tensor(0.).to(device)
                        loss_sigreg = torch.tensor(0.).to(device)

                        if test and n_student_draws > 0:
                            if k>1:
                                for j in range(k): # seeds loop
                                    for i in range(n_student_draws):           
                                        loss_sigreg += sigreg(output_s[i,:,j,:].float(), global_step) # !! TEST diversité sur les seeds
                                    if not strict_global_step:
                                        global_step += 1 # !!! TEST !!!
                            else:
                                assert False # not consistent

                        if n_student_draws > 0:
                            for i in range(n_student_draws):
                                z_draw = seeds_mlp(output_s[i].view(batch_size, k*embed_dim))
                                if stop_gradient:
                                    loss_jepa += mse(z_draw, z_center.detach())
                                else:
                                    loss_jepa += mse(z_draw, z_center) 
                                
                                if not test3:
                                    loss_sigreg += sigreg(z_draw.float(), global_step) ## !! TEST diversité sur les draws
                                    if not strict_global_step:
                                        global_step += 1

                        if strict_global_step:
                            global_step += 1


                        #loss_sigreg += sigreg(centers.view(batch_size, k*embed_dim).float(), global_step)
                        #global_step += 1

                        if k>1:
                            for j in range(k):
                                output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                                preds = output_t_seed.argmax(dim=1)
                                seeds_correct[j] += (preds == labels).sum().item()

                        #output_t_head = linear_head(z_center.detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                        output_t_head = linear_head(centers.view(batch_size, k * embed_dim).detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())

                        loss_label = criterion(output_t_head, labels)

                        loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos

                    if n_val == 0:
                        ratio = lam * loss_sigreg.item() / ((1 - lam) * loss_jepa.item() + 1e-8)
                        print(f"ratio sigreg/jepa = {ratio:.2f}")
                        print(f"pos target : ({x_probe[0].item():.2f},{y_probe[0].item():.2f}), pos_pred ({pos_pred[0,0].item():.2f},{pos_pred[0,1].item():.2f}) ")
                        print(f"avg position error = {np.sqrt(loss_pos.item()):.2f}")
                        
                        if cross_integration:
                            if k>1:
                                for i in range(k):
                                    print(f"seed {i}", mem_w[i][0,...].detach().float().cpu().numpy())
                                #print("global", w[0,...].detach().float().cpu().numpy())
                            else:
                                print(w[0,...].detach().float().cpu().numpy())
                    
                    preds = output_t_head.argmax(dim=1)
                    #print(preds)

                    correct += (preds == labels).sum().item()
                    running_sigreg += loss_sigreg.item()
                    running_jepa += loss_jepa.item()
                    running_label += loss_label.item()
                    running_pos += loss_pos.item()

                    total += labels.size(0)

            print(f"Global accuracy: {100 * correct / total:.2f}%")
            for j in range(k):
                print(f"Seed {j} accuracy : {100 * seeds_correct[j] / total:.2f}%")

            history["classif"].append(100 * correct / total)
            for j in range(k):
                history[f"classif {j}"].append(100 * seeds_correct[j] / total)
            history["loss_sigreg"].append(running_sigreg / total)
            history["loss_jepa"].append(running_jepa / total)
            history["loss_label"].append(running_label / total)
            history["loss_pos"].append(running_pos / total)
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            ist_transformer.train()
            linear_head.train()
            draws_attention.train()
            seeds_mlp.train()
            if k>1:
                heads_per_seed.train()

    if epoch % 10 == 9:
        torch.save({
                "epoch": epoch,
                "history": history,
                "ist_transformer": ist_transformer.state_dict(),
                "draws_attention": draws_attention.state_dict(),
                "seeds_mlp": seeds_mlp.state_dict(),
                "linear_head": linear_head.state_dict()
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

    if schedule:
        scheduler.step()


