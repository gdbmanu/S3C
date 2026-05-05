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

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


from torchvision import datasets


from s3c.models.foveated_vit import FoveatedMultiViT, build_foveated_pos_embed
from s3c.models.heads import TriplePredictor
from s3c.data.datasets import FoveatedUpletDataset, make_dataloader

import timm

from PIL import Image

from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

# Monter le dossier distant
local=True
if local == False:
    mount_point = os.path.expanduser("~/imagenet_mount")

    try:
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(["sshfs", "dauce.e@brain-lid-004:data/Imagenet_full", mount_point, "-o", "reconnect"], check=True)
    except:
        pass

    # Ton code ici, en utilisant mount_point
    train_dir = os.path.join(mount_point, "train")
    val_dir = os.path.join(mount_point, "val")
else:
    train_dir = "~/data/Imagenet_full/train"   # Imagenet Validation set
    val_dir = "~/data/Imagenet_full/val"   # Imagenet Validation set

load_dir = "../checkpoints/checkpoints_EMA_Xattn_260416"

dual_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_dual"

save_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_triple"


epoch_teacher = 20

n_saccades_max = 20

zoom = 1.5

std = 0.5 / zoom 

n_uplet = 3

train_epochs = 10

#model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=True)

# model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True)
model_name = "vit_base_patch8_224.dino"
model_orig = timm.create_model(model_name, pretrained=True)

#model.head = nn.Identity()  # Supprimer la tête existante
model_orig.patch_embed.strict_img_size = False
norm_ref = copy.deepcopy(model_orig.norm)

#state_dict = torch.load("dino_v3_tiny_pretrained.pth")
#model.load_state_dict(state_dict, strict=False)

build_foveated_pos_embed(model_orig,
                        base_img_size=224,
                        patch_size=8,
                        mosaic_sub_patch_grid=8,   # 8 patches par sous-image (64px/8)
                        scales=(1.0, 0.5, 0.25, 0.125))

model = FoveatedMultiViT(model_orig)

if True:
    checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_teacher}.pt")  # exemple
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Vérifie les clés disponibles
    print("Clés du checkpoint :", checkpoint.keys())
    # --- Récupération des poids du teacher ---
    if "teacher" not in checkpoint:
        raise KeyError(f"Aucune clé 'teacher' trouvée dans {checkpoint_path}")

    state_dict = checkpoint["teacher"]


    # --- Chargement dans le modèle ---
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

    print("➡️ Poids chargés (teacher).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)


model.model.norm = norm_ref # normalisation en sortie
model.to(device)
model.eval()

# 2. Geler le backbone
for param in model.parameters():
    param.requires_grad = False

train_dataset_raw   = datasets.ImageFolder(train_dir, transform=None)
val_dataset_raw   = datasets.ImageFolder(val_dir, transform=None)

train_loader = make_dataloader(train_dataset_raw, zoom, std, n_uplet, batch_size=batch_size, num_workers=num_workers, limit=None)
val_loader = make_dataloader(val_dataset_raw, zoom, std, n_uplet, batch_size=batch_size, num_workers=num_workers, limit=None)

# Définir la tête duale
triple_predictor = TriplePredictor(model_orig.embed_dim, 256)

if True:
    checkpoint_path = os.path.join(dual_dir, f"checkpoint_epoch{train_epochs}.pt")  # exemple
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
    lr=1e-4,              #
    weight_decay=1e-4, 
)

scaler = torch.cuda.amp.GradScaler()

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()

# %%

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=train_epochs - 5)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

#mosaics, sxs, sys_, zooms, labels = next(iter(train_loader)) # TEST!
#c = input('continuer?')

# %%
from tqdm import tqdm

log_interval = 100

history = {"epoch": [], "batch": [], 
        "loss": [], "loss_shift" : [], "loss_label" : [], "classif": []} #, "train_loss_3": []}

os.makedirs(save_dir, exist_ok=True)

for epoch in range(train_epochs):  # 20-30 époques suffisent

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (views, sxs, sys_, zooms, labels) in enumerate(pbar):

        views = views.to(device)          # (B, V, 3, H, W)
        sxs     = sxs.to(device)             # (B, V)
        sys_    = sys_.to(device)
        labels   = labels.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            shift1 = torch.stack([sxs[:,1] - sxs[:,0], sys_[:,1] - sys_[:,0]], dim=1)
            shift2 = torch.stack([sxs[:,2] - sxs[:,1], sys_[:,2] - sys_[:,1]], dim=1)
            shift3 = torch.stack([sxs[:,0] - sxs[:,2], sys_[:,0] - sys_[:,2]], dim=1)
            with torch.inference_mode():
                features = model(views)  # Extraction des features
            pred_shift1, pred_shift2, pred_shift3, output = triple_predictor(features[:, 0,:], features[:, 1,:], features[:, 2,:])
            loss_shift = mse(pred_shift1, shift1) + mse(pred_shift2, shift2) + mse(pred_shift3, shift3)
            loss_label = criterion(output, labels)
            loss = loss_shift + loss_label

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
                    views, sxs, sys_, zooms, labels = next(val_iter)

                    views = views.to(device)          # (B, V, 3, H, W)
                    sxs     = sxs.to(device)             # (B, V)
                    sys_    = sys_.to(device)
                    labels   = labels.to(device)

                    shift1 = torch.stack([sxs[:,1] - sxs[:,0], sys_[:,1] - sys_[:,0]], dim=1)
                    shift2 = torch.stack([sxs[:,2] - sxs[:,1], sys_[:,2] - sys_[:,1]], dim=1)
                    shift3 = torch.stack([sxs[:,0] - sxs[:,2], sys_[:,0] - sys_[:,2]], dim=1)
                    with torch.inference_mode():
                        features = model(views)  # Extraction des features
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

            torch.save({
                    "epoch": epoch,
                    "history": history,
                    "triple_predictor": triple_predictor.state_dict(),
                },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
            
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

            triple_predictor.train()

    scheduler.step()

# Démonter le dossier à la fin
subprocess.run(["fusermount", "-u", mount_point], check=True)
 

