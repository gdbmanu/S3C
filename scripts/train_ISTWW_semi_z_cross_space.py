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


from s3c.models.heads import IterativeSeedTransformerwithQuery, WhatWherePosIterativeSeedTransformer, AttentionPooling #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.utils.training import sigreg, vicReg_seed #SIGReg
from s3c.models.heads import  ABMILPosPredictor, ABMILLabelPredictor, PosPredictor, ABMILSeedProjector
from s3c.utils.training import get_parent_synset

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

k = 3 #12       # n_seeds
n_heads = 12

n_saccades_max = 30 
n_uplet_student = 3
n_uplet_teacher = 5
n_student_draws = 4
n_teacher_draws = 3
n_probes = 3

orig = False
grid = False
curriculum = False
finetune = False

if grid:
    n_saccades_max = 121
    n_uplet_student = 9
    n_uplet_teacher = 18
    n_student_draws = 6
    n_teacher_draws = 3
if finetune:
    n_student_draws = 0
    n_teacher_draws = 1


train_epochs = 100
lam = 0.05           # λ : trade-off JEPA / SIGReg
mu = 1               # spatial probe weight

supervised = True # **label** supervised
pos_supervised = True # !!
pos_separation = True # WWP !!
if supervised or pos_supervised:
    pure = False
    if False: #train_epochs == 100:
        alpha = 3e-7
    else:
        alpha = 1e-6 #3e-7
    beta = 3e-4 # !!!!! 1e-4 #3e-5 #!! 
else:
    pure = False


inv_temp = 1
stop_gradient = False

test = True # seed diversity
center_test = False # center seed consistency
vicreg = False # more seed diversity
test3 = False # no sample diversity
if not test or k==1:
    strict_global_step = True # False  #                      *** !!!!! ***
else:
    strict_global_step = False
cross_integration = True # cross_draws_integration
if finetune:
    cross_integration = False

use_synset_embeddings =  True #False # True
index_embeddings = True # False # True
synset_level = 4
if supervised or pos_supervised:
    if use_synset_embeddings:
        label_smoothing = 0.5
    else:
        label_smoothing = 0.8
else:
    label_smoothing = 0.5

abmil_pos = False #True
abmil_label = False
abmil_seed = True


suffix = ""
if supervised or pos_supervised: 
    if pure: suffix = suffix + "_PURESUP"        
    elif supervised: suffix = suffix + "_SUP"
    suffix = suffix + f"_a{alpha}"
    if beta != 1e-4 : suffix = suffix + f"_b{beta}"
    label_mask = 0.8
else:
    pure = False
    label_mask = 0.2
if pos_supervised:
    suffix = suffix + "_POS_SUP"

if test : 
    if not pure: suffix = suffix + "_TEST"
if center_test : suffix = suffix + "_CENTER"

if vicreg: suffix = suffix + "_VICREG"
if test3 : suffix = suffix + "_TEST3"
if strict_global_step : suffix = suffix + "_STRICT"
if cross_integration : suffix = suffix + "_CROSS"
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

if pos_separation: suffix = suffix + '_SEPAR'

if abmil_seed: suffix = suffix + '_ASEED'
if abmil_pos : suffix = suffix + "_APOS2"
if abmil_label : suffix = suffix + "_ALAB2"

if orig: suffix = suffix + "_ORIG"

save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_ISTWW_{k}_semi_z_lam{lam}_mu_{mu}_sab{n_sab}_LeJ{suffix}_s{n_uplet_student}_t{n_uplet_teacher}_space"

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

    
    ist_transformer = WhatWherePosIterativeSeedTransformer(n_heads=n_heads, n_seeds=k, n_blocks=n_sab, pretrained_embeddings=emb,
                                                    n_classes=n_synsets, pos_separation = pos_separation, 
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
    
    ist_transformer = WhatWherePosIterativeSeedTransformer(n_heads=n_heads, n_seeds=k, n_blocks=n_sab, pretrained_embeddings=emb,
                                                    pos_separation = pos_separation,
                                                    label_smoothing=label_smoothing, label_mask=label_mask)


draws_attention = nn.ModuleList([AttentionPooling(embed_dim, inv_temp=inv_temp) for seed_idx in range(k+2+n_probes)])


if abmil_pos:
    pos_predictor = ABMILPosPredictor(embed_dim, k)
else:
    pos_predictor = nn.Sequential(
                nn.LayerNorm(embed_dim),  
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
            )

z_pos_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),  
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1), # increase label embedding entropy
        )

# LINEAR PROBE

if abmil_label:
    linear_head = ABMILLabelPredictor(emb_dim=embed_dim, k=k)
else:
    '''linear_head = nn.Sequential(
                    nn.Unflatten(1, (k, embed_dim)),          # (B, k*d) → (B, k, d)
                    nn.LayerNorm(embed_dim),                  # norm par seed ✓
                    nn.Flatten(1),    
                    nn.Linear(k * embed_dim, 1000),
                )'''
    linear_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),                  
                    nn.Linear(embed_dim, 1000),
                )

if abmil_seed:
    seeds_mlp = ABMILSeedProjector()
else:
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

if True: #k>1:                                     # cross-draws integration (seed diversity)
    # LINEAR PROBES
    heads_per_seed = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 1000) 
            ) for _ in range(k)
        ])

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

    if "z_pos_predictor" not in checkpoint:
        raise KeyError(f"Aucune clé 'z_pos_predictor' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["z_pos_predictor"]
    missing, unexpected = z_pos_predictor.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (z_pos_predictor).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    if "seeds_mlp" not in checkpoint:
        raise KeyError(f"Aucune clé 'seeds_mlp' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["seeds_mlp"]
    missing, unexpected = seeds_mlp.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (seeds_mlp).")
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

draws_attention.to(device)
draws_attention.train()

# LINEAR PROBE

linear_head.to(device)
linear_head.train()   

pos_predictor.to(device)
pos_predictor.train()

z_pos_predictor.to(device)
z_pos_predictor.train()

seeds_mlp.to(device)
seeds_mlp.train()


if True: #k>1:
    heads_per_seed.to(device)
    heads_per_seed.train()

os.makedirs(save_dir, exist_ok=True)

if supervised or pos_supervised:
    if finetune:
        if train_epochs == 100:
            linear_optimizer = torch.optim.AdamW([
                {'params': z_pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
                weight_decay=3e-4, #0.04,  
            )            
        else:
            linear_optimizer = torch.optim.AdamW([
                {'params': z_pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
                weight_decay=1e-3, #0.04,  
            )
        ist_transformer.requires_grad_(False)
    else:
        if train_epochs == 100:
            linear_optimizer = torch.optim.AdamW(
                [{'params': ist_transformer.parameters(), 'lr': 1e-5},
                {'params': draws_attention.parameters(),       'lr': 3e-5}, #1e-5},
                {'params': z_pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': seeds_mlp.parameters(),       'lr': 3e-5}, # !!!!! 1e-4}, #1e-5},
                {'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
                weight_decay=3e-4, #0.04,  
            )
        else:
            linear_optimizer = torch.optim.AdamW(
                [{'params': ist_transformer.parameters(), 'lr': 3e-5}, #3e-6},
                {'params': draws_attention.parameters(),       'lr': 1e-4}, #1e-5},
                {'params': z_pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': pos_predictor.parameters(),       'lr': beta}, #1e-5},
                {'params': seeds_mlp.parameters(),       'lr': 1e-4}, # !!!!!! 3e-4}, #1e-5},
                {'params': linear_head.parameters(), 'lr': alpha}], #1e-4}],
                weight_decay=1e-3, #0.04,  
            )
else:
    if curriculum:
        optimizer = torch.optim.AdamW([
            {'params': ist_transformer.parameters(), 'lr': 1e-5}, #3e-6},
            {'params': draws_attention.parameters(),       'lr': 3e-5}, #1e-5},
            {'params': z_pos_predictor.parameters(),       'lr': 3e-5}, #1e-5},
            {'params': pos_predictor.parameters(),       'lr': 3e-5}, #1e-5},
            {'params': seeds_mlp.parameters(),       'lr': 1e-4}, #1e-5},
            ], #1e-5},
            weight_decay=3e-4, #0.04,  
        )
    else:
        optimizer = torch.optim.AdamW([
            {'params': ist_transformer.parameters(), 'lr': 3e-5},
            {'params': draws_attention.parameters(),       'lr': 1e-4}, #1e-5},
            {'params': z_pos_predictor.parameters(),       'lr': 3e-4}, # !!!!! 1e-4}, #3e-4},
            {'params': pos_predictor.parameters(),       'lr': 3e-4}, # !!!!! 1e-4}, #3e-4},
            {'params': seeds_mlp.parameters(),       'lr': 1e-4}, # !!!!! 3e-4},
        ], weight_decay=1e-3)

    linear_optimizer = torch.optim.AdamW(
        linear_head.parameters(),
        lr=1e-4,              #
        weight_decay=1e-3, #0.04,  
    )

if True: #k > 1:
    seeds_optimizer = torch.optim.AdamW(
        heads_per_seed.parameters(),
        lr=1e-4,              #
        weight_decay=1e-3, #0.04,  
    )

#scaler = torch.cuda.amp.GradScaler()
if label_smoothing:
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
else:
    criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

#sigreg = SIGReg()

schedule = True
if train_epochs > 30:
    n_warm = train_epochs // 6
else:
    n_warm = 5
if schedule:
    if supervised or pos_supervised:
        warmup = LinearLR(linear_optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warm)
        cosine = CosineAnnealingLR(linear_optimizer, T_max=train_epochs - n_warm)
        scheduler = SequentialLR(linear_optimizer, schedulers=[warmup, cosine], milestones=[n_warm])
    else:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=n_warm)
        cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - n_warm)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warm])


# %%

log_interval = 100

history = {"epoch": [], "batch": [], "loss": [],
        "loss_label": [], "loss_jepa" : [], "loss_sigreg" : [], "loss_pos_true": [], "loss_pos_ref": [], "loss_z_pos_ref": [], 
        "loss_pos": [], "loss_z_pos": [], "loss_pos_sup": [], "loss_z_pos_sup": []}
for j in range(k):
    history[f"classif {j}"] = []
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

        b_student = n_uplet_student * n_student_draws
        idx_s = perms[:, :b_student]    
        b_teacher = b_student + n_uplet_teacher*n_teacher_draws
        idx_t = perms[:, b_student:b_teacher] 
        b_probes = b_teacher + n_probes
        idx_probes = perms[:, b_teacher:b_probes] 
        assert b_probes <= n_saccades_max

        features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
        features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

        x_probes = sxs[torch.arange(batch_size).unsqueeze(1), idx_probes].to(device)
        y_probes = sys_[torch.arange(batch_size).unsqueeze(1), idx_probes].to(device)
        probe_targets = torch.stack([x_probes, y_probes], dim=2)                    # (B, n_probes, 2) 
        z_probes = features[torch.arange(batch_size).unsqueeze(1), idx_probes, :].to(device)

        if use_synset_embeddings:
            mem_labels = labels
            labels = label_to_synset_tensor[labels]   # (B,) — conversion immédiate

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            detach_label = not supervised
            if supervised or pos_supervised:
                if n_student_draws > 0:
                    output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:], labels, probe_targets, detach_label=detach_label) for i in range(n_student_draws)], dim=1)
                output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:], labels, probe_targets, detach_label=detach_label) for i in range(n_teacher_draws)], dim=1)
            else:
                if n_student_draws > 0:
                    output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:], None, probe_targets) for i in range(n_student_draws)], dim=1)
                output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:], None, probe_targets) for i in range(n_teacher_draws)], dim=1)

            if use_synset_embeddings:
                labels = mem_labels

            if cross_integration:
                center_seeds = []
                for seed_idx in range(k+2+n_probes):
                    # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                    z, _ = draws_attention[seed_idx](output_t[:, :, seed_idx, :])           # (B, d)
                    center_seeds.append(z)
                centers = torch.stack(center_seeds, dim=1)
            else:
                centers = output_t.mean(dim=1)

            seed_centers = centers[:,:k,:]
            z_center, _ = seeds_mlp(seed_centers) #.view(batch_size, k*embed_dim))
            
            z_label_center = centers[:,k,:]

            if abmil_pos:
                z_pos_centers = z_pos_predictor(centers[:,(k+1):,:]) #.view(batch_size, embed_dim)
            else:
                z_pos_centers = z_pos_predictor(centers[:,(k+2):,:]) #.view(batch_size, embed_dim)
            # z_pos_draws = output_t[:, :, k, :] #.view(batch_size, embed_dim)

            loss_jepa = torch.tensor(0.).to(device)
            loss_sigreg = torch.tensor(0.).to(device)

            if test and n_student_draws > 0:   ### SEED DIVERSITY ###
                if k>1:
                    for j in range(k): # seeds loop
                        for i in range(n_student_draws):           
                            loss_sigreg += sigreg(output_s[:,i,j,:].float(), global_step) # !! TEST diversité sur les seeds                            
                        if not strict_global_step:
                            global_step += 1 # !!! TEST !!!

            if n_student_draws > 0:
                z_draws = []
                for i in range(n_student_draws):
                    z_draw, _ = seeds_mlp(output_s[:,i,:k,:]) #.view(batch_size, k*embed_dim))
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

            ### LOSS_Z_POS : z_star + z_probes ###
            if pos_supervised:
                if abmil_pos:
                    z_pos_targets = torch.cat([z_star.unsqueeze(1), z_probes], dim = 1)
                else:
                    z_pos_targets = z_probes
                loss_z_pos = F.mse_loss(z_pos_centers, z_pos_targets)
            else:
                z_pos_targets = z_probes
                loss_z_pos = F.mse_loss(z_pos_centers[:,1:], z_pos_targets)

                   
            ### LOSS_POS : (x,y) positions ###
            if pos_supervised:
                pos_target = torch.stack([x_star, y_star], dim=1)                           # (B, 2) 
            else:
                pos_target = torch.zeros((batch_size, 2), device=device)
            
            if abmil_pos:
                pos_targets = torch.cat([pos_target.unsqueeze(1), probe_targets], dim=1)    # (B, n_probes + 1, 2)
            else:
                pos_targets = pos_target

            if abmil_pos:   
                pos_preds, _ = pos_predictor(seed_centers.clone().detach(), z_pos_targets) # !!! PAS z_pos_center !!!             
            else:
                pos_preds = pos_predictor(centers[:,k+1,:]) #z_pos_targets) # !!! PAS z_pos_center !!!
            loss_pos = F.mse_loss(pos_preds, pos_targets)
            
           
            ### LABEL LOSS
            if supervised or pos_supervised:
                if abmil_label:
                    output_t_head, _ = linear_head(seed_centers, z_label_center) # !!! 
                else:
                    output_t_head = linear_head(z_label_center) #seed_centers.view(batch_size, k * embed_dim)) #
                loss_label = criterion(output_t_head, labels)
                
                if pure:
                    loss = loss_label 
                else:
                    loss_label = loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos + loss_z_pos + loss_label #criterion(output_t_head, labels)
            else:
                if abmil_label:
                    output_t_head, _ = linear_head(seed_centers.detach(), z_label_center.detach())
                else:
                    output_t_head = linear_head(z_label_center.detach()) #linear_head(seed_centers.view(batch_size, k * embed_dim).detach()) #linear_head(centers.view(batch_size, k * embed_dim).detach()) #linear_head(output_t[0].detach()) + linear_head(output_t[1].detach())
                    #output_t_head = linear_head(z_center.detach()) 
                loss_label = criterion(output_t_head, labels)
                loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos + loss_z_pos # !!!

            ### SEEDS label
            loss_seeds = 0
            for j in range(k):
                output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                loss_seeds += criterion(output_t_seed, labels)          


        if not supervised and not pos_supervised:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
            if True: #k>1:
                torch.nn.utils.clip_grad_norm_(draws_attention.parameters(), 1.0)
            optimizer.step()

        linear_optimizer.zero_grad()
        loss_label.backward()
        if supervised or pos_supervised:
            torch.nn.utils.clip_grad_norm_(ist_transformer.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pos_predictor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(z_pos_predictor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(draws_attention.parameters(), 1.0)
        linear_optimizer.step()

        if True: #k>1:
            seeds_optimizer.zero_grad()
            loss_seeds.backward()
            seeds_optimizer.step()

        total_loss += loss.item()
    
        if (batch_idx + 1) % log_interval == 0:

            ist_transformer.eval()
            linear_head.eval()
            pos_predictor.eval()
            z_pos_predictor.eval()
            draws_attention.eval()
            seeds_mlp.eval()                
            if True: #k>1:
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
            running_pos_ref = 0.0
            running_pos_true = 0.0
            running_z_pos_ref = 0.0
            running_pos = 0.0
            running_z_pos = 0.0
            running_pos_sup = 0.0
            running_z_pos_sup = 0.0
            seeds_correct = [0.0 for j in range(k)]
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

                    b_student = n_uplet_student * n_student_draws
                    idx_s = perms[:, :b_student]    
                    b_teacher = b_student + n_uplet_teacher * n_teacher_draws
                    idx_t = perms[:, b_student:b_teacher] 
                    b_probes = b_teacher + n_probes
                    idx_probes = perms[:, b_teacher:b_probes] 
                    assert b_probes <= n_saccades_max

                    features_s = features[torch.arange(batch_size).unsqueeze(1), idx_s, :].to(device)  # (batch_size, k, 768)
                    features_t = features[torch.arange(batch_size).unsqueeze(1), idx_t, :].to(device)  # (batch_size, k, 768)

                    x_probes = sxs[torch.arange(batch_size).unsqueeze(1), idx_probes].to(device)
                    y_probes = sys_[torch.arange(batch_size).unsqueeze(1), idx_probes].to(device)
                    probe_targets = torch.stack([x_probes, y_probes], dim=2)                            # (B, n_probes, 2) 
                    z_probes = features[torch.arange(batch_size).unsqueeze(1), idx_probes, :].to(device)

                    if use_synset_embeddings:
                        mem_labels = labels
                        labels = label_to_synset_tensor[labels]   # (B,) — conversion immédiate

                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if n_student_draws > 0:
                            output_s = torch.stack([ist_transformer(features_s[:, i*n_uplet_student : (i+1)*n_uplet_student,:], None, probe_targets) for i in range(n_student_draws)], dim=1)
                        output_t = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:], None, probe_targets) for i in range(n_teacher_draws)], dim=1)
                        output_t_sup = torch.stack([ist_transformer(features_t[:, i*n_uplet_teacher : (i+1)*n_uplet_teacher,:], labels, probe_targets) for i in range(n_teacher_draws)], dim=1)
                        
                        if use_synset_embeddings:
                            labels = mem_labels
                        mem_w = []
                        mem_w_sup = []

                        if cross_integration:
                            center_seeds = []
                            center_seeds_sup = []
                            for seed_idx in range(k+2+n_probes):
                                # Vues de ce seed à travers tous les draws : (B, n_draws, d)
                                z, w = draws_attention[seed_idx](output_t[:, :, seed_idx, :])           # (B, d)
                                mem_w += [w]
                                center_seeds.append(z)
                                z_sup, w_sup = draws_attention[seed_idx](output_t_sup[:, :, seed_idx, :])           # (B, d)
                                mem_w_sup += [w_sup]
                                center_seeds_sup.append(z_sup)
                            centers = torch.stack(center_seeds, dim=1)
                            centers_sup = torch.stack(center_seeds_sup, dim=1)
                        else:
                            centers = output_t.mean(dim=1)
                            centers_sup = output_t_sup.mean(dim=1)

                        seed_centers = centers[:,:k,:]
                        z_center, _ = seeds_mlp(seed_centers) 
                        seed_centers_sup = centers_sup[:,:k,:]
                        z_center_sup, _ = seeds_mlp(seed_centers_sup) 

                        z_label_center = centers[:,k,:]
                        z_pos_centers = z_pos_predictor(centers[:,(k+1):,:])

                        z_label_center_sup = centers_sup[:,k,:]
                        z_pos_centers_sup = z_pos_predictor(centers_sup[:,(k+1):,:])

                        loss_jepa = torch.tensor(0.).to(device)
                        loss_sigreg = torch.tensor(0.).to(device)

                        if test and n_student_draws > 0:   ### SEED DIVERSITY ###
                            if k>1:
                                for j in range(k): # seeds loop
                                    for i in range(n_student_draws):           
                                        loss_sigreg += sigreg(output_s[:,i,j,:].float(), global_step) # !! TEST diversité sur les seeds                            
                                    if not strict_global_step:
                                        global_step += 1 # !!! TEST !!!

                        if n_student_draws > 0:
                            z_draws = []
                            for i in range(n_student_draws):
                                z_draw, _ = seeds_mlp(output_s[:,i,:k,:]) #.view(batch_size, k*embed_dim))
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

                        
                        ### LOSS_Z_POS : z_star + z_probes ###
                        if pos_supervised:
                            z_pos_targets = torch.cat([z_star.unsqueeze(1), z_probes], dim = 1)
                            loss_z_pos_ref = F.mse_loss(z_pos_centers[:,1,:], z_pos_targets[:,1,:])
                            loss_z_pos = F.mse_loss(z_pos_centers[:,0,:], z_pos_targets[:,0,:])
                            loss_z_pos_sup = F.mse_loss(z_pos_centers_sup[:,0,:], z_pos_targets[:,0,:]) 
                        else:
                            z_pos_targets = z_probes
                            loss_z_pos = loss_z_pos_ref = F.mse_loss(z_pos_centers[:,1:], z_pos_targets)
                            loss_z_pos_sup = torch.tensor(0.).to(device) ## ?? TODO ??

                        ### LOSS_POS : (x,y) positions ###
                        if pos_supervised:
                            pos_target = torch.stack([x_star, y_star], dim=1)                           # (B, 2) 
                        else:
                            pos_target = torch.zeros((batch_size, 2), device=device)
                        
                        pos_targets = torch.cat([pos_target.unsqueeze(1), probe_targets], dim=1)    # (B, n_probes + 1, 2)  
                        
                        if abmil_pos:
                            pos_preds_true, _ = pos_predictor(seed_centers, z_pos_targets) 
                            pos_preds, _ = pos_predictor(seed_centers, z_pos_centers) 
                            pos_preds_sup, _ = pos_predictor(seed_centers, z_pos_centers_sup)
                        else:
                            pos_preds_true = pos_predictor(centers[:,k+1,:]).unsqueeze(1) #z_pos_targets)     
                            pos_preds = pos_predictor(centers[:,k+1,:]).unsqueeze(1) #z_pos_centers)     
                            pos_preds_sup = pos_predictor(centers_sup[:,k+1,:]).unsqueeze(1) #z_pos_centers_sup)
                        
                        if abmil_pos:
                            loss_pos_ref = F.mse_loss(pos_preds[:,1,:], pos_targets[:,1,:]) 
                            loss_pos_true = F.mse_loss(pos_preds_true[:,1,:], pos_targets[:,1,:]) 
                        else:
                            loss_pos_ref = torch.tensor(0.).to(device)
                            loss_pos_true = torch.tensor(0.).to(device)
                        loss_pos = F.mse_loss(pos_preds[:,0,:], pos_targets[:,0,:])
                        loss_pos_sup = F.mse_loss(pos_preds_sup[:,0,:], pos_targets[:,0,:])

                        ### LABEL LOSS
                        if abmil_label:
                            output_t_head, _ = linear_head(seed_centers, z_label_center)
                            output_t_head_sup, _ = linear_head(seed_centers, z_label_center_sup)
                        else:
                            output_t_head = linear_head(z_label_center) #linear_head(seed_centers.view(batch_size, k * embed_dim)) #seed_centers.view(batch_size, k * embed_dim))
                            output_t_head_sup = linear_head(z_label_center_sup) #linear_head(seed_centers.view(batch_size, k * embed_dim)) #seed_centers.view(batch_size, k * embed_dim))
                            #output_t_head = linear_head(z_center) #linear_head(seed_centers.view(batch_size, k * embed_dim)) #seed_centers.view(batch_size, k * embed_dim))
                            #output_t_head_sup = linear_head(z_center_sup)
                        if supervised or pos_supervised:
                            loss_label = loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos + loss_z_pos + criterion(output_t_head, labels)
                        else:
                            loss_label = criterion(output_t_head, labels)
                            loss = (1 - lam) * loss_jepa + lam * loss_sigreg + loss_pos + loss_z_pos

                    ### SEEDS label
                    loss_seeds = 0
                    for j in range(k):
                        output_t_seed = heads_per_seed[j](centers[:,j,:].detach())
                        seeds_pred = output_t_seed.argmax(dim=1) 
                        seeds_correct[j] += (seeds_pred == labels).sum().item()         


                    if n_val == 0:
                        ratio = lam * loss_sigreg.item() / ((1 - lam) * loss_jepa.item() + 1e-8)
                        print(f"ratio sigreg/jepa = {ratio:.2f}")
                        if pos_supervised:
                            print(f"pos target : ({x_star[0].item():.3f},{y_star[0].item():.3f}), pos_pred ({pos_preds[0,0,0].item():.3f},{pos_preds[0,0,1].item():.3f}), pos_pred_sup ({pos_preds_sup[0,0,0].item():.3f},{pos_preds_sup[0,0,1].item():.3f}) ")
                            if abmil_pos:
                                print(f"pos probe : ({x_probes[0,0].item():.3f},{y_probes[0,0].item():.3f}), pos_pred ({pos_preds[0,1,0].item():.3f},{pos_preds[0,1,1].item():.3f}), pos_pred_sup ({pos_preds_sup[0,1,0].item():.3f},{pos_preds_sup[0,1,1].item():.3f}) ")
                        else:
                            print(f"pos target : ({x_probes[0,0].item():.3f},{y_probes[0,0].item():.3f}), pos_pred ({pos_preds[0,1,0].item():.3f},{pos_preds[0,1,1].item():.3f}) ")
                        print(f"z_pos ref error = {np.sqrt(loss_z_pos_ref.item()):.3f}")
                        print(f"z_pos error = {np.sqrt(loss_z_pos.item()):.3f}")
                        if pos_supervised:
                            print(f"z_pos sup error = {np.sqrt(loss_z_pos_sup.item()):.3f}")   
                        print(f"pos ref error = {np.sqrt(loss_pos_ref.item()):.3f}")
                        print(f"pos true error = {np.sqrt(loss_pos_true.item()):.3f}")
                        print(f"pos error = {np.sqrt(loss_pos.item()):.3f}")
                        if pos_supervised:
                            print(f"pos sup error = {np.sqrt(loss_pos_sup.item()):.3f}")
                        
                        if cross_integration:
                            if True: #k>1:
                                for i in range(k):
                                    print(f"seed {i}", mem_w[i][0,...].detach().float().cpu().numpy().flatten())
                                print(f"pos", mem_w[k][0,...].detach().float().cpu().numpy().flatten())
                                print(f"pos_sup", mem_w_sup[k][0,...].detach().float().cpu().numpy().flatten())
                            else:
                                print(w[0,...].detach().float().cpu().numpy())
                    
                    preds = output_t_head.argmax(dim=1)
                    #print(preds)

                    correct += (preds == labels).sum().item()
                    running_sigreg += loss_sigreg.item()
                    running_jepa += loss_jepa.item()
                    running_label += loss_label.item()
                    running_pos_ref += loss_pos_ref.item()
                    running_pos_true += loss_pos_true.item()
                    running_z_pos_ref += loss_z_pos_ref.item()
                    running_pos += loss_pos.item()
                    running_z_pos += loss_z_pos.item()
                    running_pos_sup += loss_pos_sup.item()
                    running_z_pos_sup += loss_z_pos_sup.item()

                    preds_sup = output_t_head_sup.argmax(dim=1)
                    correct_sup += (preds_sup == labels).sum().item()

                    total += labels.size(0)

            print(f"Global accuracy: {100 * correct / total:.2f}%")
            print(f"Oracle accuracy: {100 * correct_sup / total:.2f}%")
            for j in range(k):
                print(f"Seed {j} accuracy : {100 * seeds_correct[j] / total:.2f}%")

            history["classif"].append(100 * correct / total)
            for j in range(k):
                history[f"classif {j}"].append(100 * seeds_correct[j] / total)
            history[f"sup classif"].append(100 * correct_sup / total)
            history["loss_sigreg"].append(running_sigreg / total)
            history["loss_jepa"].append(running_jepa / total)
            history["loss_label"].append(running_label / total)
            history["loss_pos_ref"].append(running_pos_ref / total)
            history["loss_pos_true"].append(running_pos_true / total)
            history["loss_z_pos_ref"].append(running_z_pos_ref / total)
            history["loss_pos"].append(running_pos / total)
            history["loss_z_pos"].append(running_z_pos / total)
            history["loss_pos_sup"].append(running_pos_sup / total)
            history["loss_z_pos_sup"].append(running_z_pos_sup / total)
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            ist_transformer.train()
            linear_head.train()
            pos_predictor.train()
            z_pos_predictor.train()
            draws_attention.train()
            seeds_mlp.train()
            if True: #k>1:
                heads_per_seed.train()

    if epoch % 10 == 9:
        torch.save({
                "epoch": epoch,
                "history": history,
                "ist_transformer": ist_transformer.state_dict(),
                "draws_attention": draws_attention.state_dict(),
                "seeds_mlp": seeds_mlp.state_dict(),
                "linear_head": linear_head.state_dict(),
                "pos_predictor": pos_predictor.state_dict(),
                "z_pos_predictor": z_pos_predictor.state_dict(),
                "heads_per_seed": heads_per_seed.state_dict(), # if True #k>1 else None
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

    if schedule:
        scheduler.step()


