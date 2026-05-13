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

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from torchvision import datasets


from s3c.models.heads import TriplePredictor
from s3c.data.datasets import ImageNetZDataset

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

dual_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_dual"

save_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_triple_z"


epoch_teacher = 20

n_saccades_max = 30

zoom = 1.5

std = 0.5 / zoom 

n_uplet = 3

train_epochs_head = 10

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

# Définir la tête triple
triple_predictor = TriplePredictor(embed_dim, 256)

if True:
    checkpoint_path = os.path.join(dual_dir, f"checkpoint_epoch{train_epochs_head}.pt")  # exemple
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Vérifie les clés disponibles
    print("Clés du checkpoint :", checkpoint.keys())
    # --- Récupération des poids du teacher ---
    if "dual_predictor" not in checkpoint:
        raise KeyError(f"Aucune clé 'dual_predictor' trouvée dans {checkpoint_path}")

    state_dict = checkpoint["dual_predictor"]


    # --- Chargement dans le modèle ---
    missing, unexpected = triple_predictor.load_state_dict(state_dict, strict=False)

    print("➡️ Poids chargés (dual_predictor).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)


triple_predictor.to(device)
triple_predictor.train()

os.makedirs(save_dir, exist_ok=True)

# Optimiseur
#optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.001, momentum=0.9)

optimizer = torch.optim.AdamW(
    triple_predictor.parameters(),
    lr=1e-6,              #
    weight_decay=1e-3, #1e-4, 
)

scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()


schedule = True
if schedule:
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])


# %%
from tqdm import tqdm

log_interval = 100

history = {"epoch": [], "batch": [], 
        "loss": [], "loss_shift" : [], "loss_label" : [], "classif": []} #, "train_loss_3": []}

os.makedirs(save_dir, exist_ok=True)

for epoch in range(train_epochs):  

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (features, sxs, sys_, labels) in enumerate(pbar):

        features = features.to(device)          # (B, n_saccades_max, 768)
        sxs     = sxs.to(device)             # (B, n_saccades_max)
        sys_    = sys_.to(device)
        labels   = labels.to(device)

        # Génère des indices aléatoires pour chaque échantillon du batch
        # Shape : (batch_size, 3)
        batch_size, n = sxs.shape
        random_indices = torch.stack([torch.randperm(n_saccades_max)[:3] for _ in range(batch_size)])

        # Sélectionne les 3 valeurs pour x, y et z
        features = features[torch.arange(batch_size).unsqueeze(1), random_indices, :]  # (batch_size, 3, 768)
        sxs = sxs[torch.arange(batch_size).unsqueeze(1), random_indices]  # (batch_size, 3)
        sys_ = sys_[torch.arange(batch_size).unsqueeze(1), random_indices]  # (batch_size, 3)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            shift1 = torch.stack([sxs[:,1] - sxs[:,0], sys_[:,1] - sys_[:,0]], dim=1)
            shift2 = torch.stack([sxs[:,2] - sxs[:,1], sys_[:,2] - sys_[:,1]], dim=1)
            shift3 = torch.stack([sxs[:,0] - sxs[:,2], sys_[:,0] - sys_[:,2]], dim=1)
            pred_shift1, pred_shift2, pred_shift3, output = triple_predictor(features[:, 0,:], features[:, 1,:], features[:, 2,:])
            loss_shift = mse(pred_shift1, shift1) + mse(pred_shift2, shift2) + mse(pred_shift3, shift3)
            loss_label = criterion(output, labels)
            loss = 0.1 * loss_shift + loss_label

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    


        if (batch_idx + 1) % log_interval == 0:

            triple_predictor.eval()

            print(f"Epoch {epoch+1:03d} | simple loss = {total_loss / log_interval:.4f}")

            history["epoch"].append(epoch + 1)
            history["batch"].append(batch_idx + 1)
            history["loss"].append(total_loss / log_interval)

            total_loss = 0

            total = 0
            correct = 0.0
            running_shift= 0.0
            running_label = 0.0
            val_iter = iter(val_loader)

            with torch.no_grad():
                for _ in range(5):
                    features, sxs, sys_, labels = next(val_iter)

                    batch_size, n = sxs.shape

                    features = features.to(device)          # (B, n_saccades_max, 768)
                    sxs     = sxs.to(device)             # (B, n_saccades_max)
                    sys_    = sys_.to(device)
                    labels   = labels.to(device)

                    # Génère des indices aléatoires pour chaque échantillon du batch
                    # Shape : (batch_size, 3)
                    random_indices = torch.stack([torch.randperm(n_saccades_max)[:3] for _ in range(batch_size)])

                    # Sélectionne les 3 valeurs pour x, y et z
                    features = features[torch.arange(batch_size).unsqueeze(1), random_indices, :]  # (batch_size, 3, 768)
                    sxs = sxs[torch.arange(batch_size).unsqueeze(1), random_indices]  # (batch_size, 3)
                    sys_ = sys_[torch.arange(batch_size).unsqueeze(1), random_indices]  # (batch_size, 3)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        shift1 = torch.stack([sxs[:,1] - sxs[:,0], sys_[:,1] - sys_[:,0]], dim=1)
                        shift2 = torch.stack([sxs[:,2] - sxs[:,1], sys_[:,2] - sys_[:,1]], dim=1)
                        shift3 = torch.stack([sxs[:,0] - sxs[:,2], sys_[:,0] - sys_[:,2]], dim=1)
                        pred_shift1, pred_shift2, pred_shift3, output = triple_predictor(features[:, 0,:], features[:, 1,:], features[:, 2,:])
                        loss_shift = mse(pred_shift1, shift1) + mse(pred_shift2, shift2) + mse(pred_shift3, shift3)
                        loss_label = criterion(output, labels)
                        loss = loss_shift + loss_label

                    preds = output.argmax(dim=1)

                    correct += (preds == labels).sum().item()
                    running_shift += loss_shift.item()
                    running_label = loss_label.item()

                    total += labels.size(0)

            print(f"Top-1 accuracy: {100 * correct / total:.2f}%")

            history["classif"].append(100 * correct / total)
            history["loss_shift"].append(running_shift / total)
            history["loss_label"].append(running_label / total)

            triple_predictor.train()

    if epoch % 9 == 0:
        torch.save({
                "epoch": epoch,
                "history": history,
                "triple_predictor": triple_predictor.state_dict(),
            },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

    if schedule:
        scheduler.step()


