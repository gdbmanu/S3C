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


from s3c.models.heads import WhatTransformer, AttentionPooling #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.models.heads import  PosPredictor
from s3c.utils.training import get_parent_synset

import timm

from PIL import Image

from tqdm import tqdm

from datetime import datetime


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 128 #256
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 768
bottleneck_dim = 768

epoch_teacher = 20

zoom = 1.5
std = 0.5 / zoom 

n_sab = 4 #2

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

pos_supervised = True # !!
alpha = 1e-6
delta = 3e-7

inv_temp = 1
stop_gradient = False

use_synset_embeddings =  True #False # True
index_embeddings = True # False # True
synset_level = 4
if use_synset_embeddings:
    label_smoothing = 0.5
else:
    label_smoothing = 0.8
label_mask = 0.2

suffix = ""
suffix = suffix + f"_a{alpha}"
suffix = suffix + f"_d{delta}"
suffix = suffix + f'_mask{label_mask}'

if pos_supervised:
    suffix = suffix + "_POS_SUP"

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
if index_embeddings:
    suffix = suffix + "_INDEX"
if use_synset_embeddings:
    suffix = suffix + f"_SYNSET{synset_level}"

if orig: suffix = suffix + "_ORIG"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_W_sab{n_sab}_{suffix}_t{n_uplet_teacher}_space"

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

if use_synset_embeddings:

    dataset = ImageFolder(root='~/data/Imagenet_full/val')
    wnids = list(dataset.class_to_idx.keys())   # ['n01440764', ...]
    wnids = sorted(wnids)                        # ordre alphabétique = ordre ImageNet standard
    print(len(wnids))   # 1000

    label_to_parent = {}
    parent_names    = {}

    for idx, wnid in enumerate(wnids):
        parent = get_parent_synset(wnid, level=synset_level)
        parent_name = parent.lemmas()[0].name().replace('_', ' ')
        label_to_parent[idx] = parent_name
        parent_names[parent_name] = parent_names.get(parent_name, len(parent_names))

    # Mapping label_idx → indice du synset parent
    n_synsets = len(parent_names)
    label_to_synset_idx = {
        idx: parent_names[name]
        for idx, name in label_to_parent.items()
    }

    print(f"1000 classes → {n_synsets} synsets de niveau {synset_level}")
    # Tensor de mapping pour usage GPU
    label_to_synset_tensor = torch.tensor(
        [label_to_synset_idx[i] for i in range(1000)],
        dtype=torch.long
    ).to(device)   # (1000,)

    # Noms des synsets ordonnés par indice
    synset_names = sorted(parent_names.keys(), key=lambda x: parent_names[x])
    print(synset_names[:100])

    model, _ = clip.load("ViT-L/14")
    model.eval().cuda()

    texts = clip.tokenize([f"a photo of a {s}" for s in synset_names]).cuda()

    with torch.no_grad():
        synset_clip_embeddings = model.encode_text(texts)   # (n_synsets, 768)

    del model
    torch.cuda.empty_cache()
    if index_embeddings:
        emb = None
    else:
        emb = synset_clip_embeddings

    torch.save({
        'embeddings':      synset_clip_embeddings.cpu(),   # (n_synsets, 768)
        'synset_names':    synset_names,               # liste de noms
        'label_to_synset': label_to_synset_tensor.cpu(), # (1000,)
        'n_synsets':       n_synsets,
    }, f'imagenet_synset_{synset_level}_embeddings.pt')

    
    ist_transformer = WhatTransformer(n_heads=n_heads, n_blocks=n_sab, pretrained_embeddings=emb,
                                                    n_classes=n_synsets, 
                                                    label_smoothing=label_smoothing, label_mask=label_mask)


else:
    classes = ResNet50_Weights.DEFAULT.meta['categories']
    print(len(classes))      # 1000
    print(classes[0])        # 'tench'
    print(classes[999])      # 'toilet tissue'
    # Pour CLIP
    texts = clip.tokenize([f"a photo of a {c}" for c in classes]).cuda()
    model, _ = clip.load("ViT-L/14")
    model.eval().cuda()
    with torch.no_grad():
        label_embeddings = model.encode_text(texts)        # (1000, 768)
    # Effacer le modèle CLIP après utilisation
    del model
    torch.cuda.empty_cache()
    if index_embeddings:
        emb = None
    else:
        emb = label_embeddings
    
    ist_transformer = WhatTransformer(n_heads=n_heads, n_blocks=n_sab, pretrained_embeddings=emb,
                                                    label_smoothing=label_smoothing, label_mask=label_mask)

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


if pos_supervised:
    z_linear_head_dir = "../checkpoints/checkpoints_260414_EMA_Xattn_1_view"

    z_linear_head = nn.Linear(embed_dim, 1000)

    checkpoint_path = os.path.join(z_linear_head_dir, f"checkpoint_epoch20.pt")  
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Vérifie les clés disponibles
    print("Clés du checkpoint :", checkpoint.keys())
    # --- Récupération des poids du teacher ---
    if "classifier" not in checkpoint:
        raise KeyError(f"Aucune clé 'classifier' trouvée dans {checkpoint_path}")

    state_dict = checkpoint["classifier"]

    # --- Chargement dans le modèle ---
    missing, unexpected = z_linear_head.load_state_dict(state_dict, strict=False)

    print("➡️ Poids chargés (z_linear_head).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    z_linear_head.to(device)
    z_linear_head.train()


ist_transformer.to(device)
ist_transformer.train()

# LINEAR PROBE

linear_head.to(device)
linear_head.train()   

os.makedirs(save_dir, exist_ok=True)

if finetune:
    linear_optimizer = torch.optim.AdamW([
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
        [{'params': ist_transformer.parameters(), 'lr': delta}], #1e-5},
        weight_decay=weight_decay, #0.04,  
    )
    linear_optimizer = torch.optim.AdamW(
        [{'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
        weight_decay=weight_decay, #0.04,  
    )
    


#scaler = torch.cuda.amp.GradScaler()
if label_smoothing:
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
else:
    criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

schedule = True
if train_epochs > 30:
    n_warm = train_epochs // 6
else:
    n_warm = 5
if schedule:
    warmup = LinearLR(linear_optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warm)
    cosine = CosineAnnealingLR(linear_optimizer, T_max=train_epochs - n_warm)
    scheduler = SequentialLR(linear_optimizer, schedulers=[warmup, cosine], milestones=[n_warm])
    


# %%

log_interval = 100

history = {"epoch": [], "batch": [], "loss": [],
        "loss_label": [],  
        "loss_z_pos": [], "loss_z_pos_sup": []}
history[f"classif"] = []
history[f"sup classif"] = []

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

        if pos_supervised:
            logits = z_linear_head(features.view(batch_size * n_saccades_max, -1))   # (B*n_sac, 1000)
            logits = logits.view(batch_size, n_saccades_max, 1000)                    # (B, n_sac, 1000)

            # logit du label correct pour chaque saccade
            correct_logits = logits[torch.arange(batch_size), :, labels]              # (B, n_sac)
            # meilleure saccade
            i_star = correct_logits.argmax(dim=1)                            # (B,)
            # coordonnées
            x_star = sxs[torch.arange(batch_size),  i_star]                          # (B,)
            y_star = sys_[torch.arange(batch_size), i_star]  
            z_star = features[torch.arange(batch_size), i_star, :]                          # (B,)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, k)
        perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

        b_teacher = n_uplet_teacher
        idx_t = perms[:, :b_teacher]

        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        if use_synset_embeddings:
            mem_labels = labels
            labels = label_to_synset_tensor[labels]   # (B,) — conversion immédiate

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_t = ist_transformer(features_t[:, :n_uplet_teacher,:], labels) 

            loss = F.mse_loss(output_t, z_star)
           
            if use_synset_embeddings:
                labels = mem_labels
           
            ### LABEL LOSS           
            output_t_head = linear_head(output_t[:,0,:].detach().clone()) 
            loss_label = criterion(output_t_head, labels)

        if not finetune:
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
            running_z_pos_sup = 0.0
            val_iter = iter(val_loader)

            if True:
                correct_sup = 0.0

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

                    if pos_supervised:
                        logits = z_linear_head(features.view(batch_size * n_saccades_max, -1))   # (B*n_sac, 1000)
                        logits = logits.view(batch_size, n_saccades_max, 1000)                    # (B, n_sac, 1000)

                        # logit du label correct pour chaque saccade
                        correct_logits = logits[torch.arange(batch_size), :, labels]              # (B, n_sac)
                        # meilleure saccade
                        i_star = correct_logits.argmax(dim=1)                            # (B,)
                        # coordonnées
                        x_star = sxs[torch.arange(batch_size), i_star]                          # (B,)
                        y_star = sys_[torch.arange(batch_size), i_star]                           # (B,)
                        z_star = features[torch.arange(batch_size), i_star, :]                           # (B,)

                    perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])

                    b_teacher = n_uplet_teacher
                    idx_t = perms[:, :b_teacher] 

                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    if use_synset_embeddings:
                        mem_labels = labels
                        labels = label_to_synset_tensor[labels]   # (B,) — conversion immédiate

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_t = ist_transformer(features_t[:, :n_uplet_teacher,:], None) 
                        output_t_sup = ist_transformer(features_t[:, :n_uplet_teacher,:], labels) 

                        loss = F.mse_loss(output_t, z_star)
                        loss_sup = F.mse_loss(output_t_sup, z_star)
                        
                        if use_synset_embeddings:
                            labels = mem_labels

                        ### LABEL LOSS           
                        output_t_head = linear_head(output_t[:,0,:]) 
                        output_t_head_sup = linear_head(output_t_sup[:,0,:]) 
                        loss_label = criterion(output_t_head, labels)
                        loss_label_sup = criterion(output_t_head_sup, labels)

                    if n_val == 0:
                        if pos_supervised:
                            print(f"z star error = {np.sqrt(loss.item()):.3f}")   
                            print(f"z star sup error = {np.sqrt(loss_sup.item()):.3f}")
                    
                    preds = output_t_head.argmax(dim=1)
                    #print(preds)

                    correct += (preds == labels).sum().item()
                    running_label += loss_label.item()
                    running_z_pos += loss.item()
                    running_z_pos_sup += loss_sup.item()

                    preds_sup = output_t_head_sup.argmax(dim=1)
                    correct_sup += (preds_sup == labels).sum().item()

                    total += labels.size(0)

            print(f"Global accuracy: {100 * correct / total:.2f}%")
            print(f"Oracle accuracy: {100 * correct_sup / total:.2f}%")

            history["classif"].append(100 * correct / total)
            history[f"sup classif"].append(100 * correct_sup / total)
            history["loss_label"].append(running_label / total)
            history["loss_z_pos"].append(running_z_pos / total)
            history["loss_z_pos_sup"].append(running_z_pos_sup / total)
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


