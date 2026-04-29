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

import timm

from s3c.models.foveated_vit import FoveatedMultiViT, build_foveated_pos_embed
from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform
from s3c.data.datasets import FoveatedUpletDataset

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

mount_point = os.path.expanduser("~/imagenet_mount")
try:
    os.makedirs(mount_point, exist_ok=True)
    subprocess.run(["sshfs", "dauce.e@brain-lid-004:data/Imagenet_full", mount_point, "-o", "reconnect"], check=True)
except:
    pass


train_dir = os.path.join(mount_point, "train")
val_dir = os.path.join(mount_point, "val")


# train_dir = "~/data/Imagenet_full/train"   # Imagenet Validation set
# val_dir = "~/data/Imagenet_full/val"   # Imagenet Validation set

load_dir = "../checkpoints/checkpoints_EMA_Xattn_260321"

save_dir = "../checkpoints/checkpoints_260414_EMA_Xattn_linear_head"
#save_dir = "./checkpoints_260414_base_1_view"


epoch_teacher = 20

zoom = 1.5

std = 0.3 #* zoom

n_uplet = 1

train_epochs = 20


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

checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_teacher}.pt")  # exemple
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# Vérifie les clés disponibles
print("Clés du checkpoint :", checkpoint.keys())
# --- Récupération des poids du teacher ---
if "teacher" not in checkpoint:
    raise KeyError(f"Aucune clé 'teacher' trouvée dans {checkpoint_path}")

state_dict = checkpoint["teacher"]

model = FoveatedMultiViT(model_orig)

# --- Chargement dans le modèle ---
if True: ## !! BASE
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

    print("➡️ Poids chargés (teacher).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)


model.model.norm = norm_ref #= nn.Identity() #!! pas de normalisation en sortie
model.to(device)
model.eval()

# 2. Geler le backbone
for param in model.parameters():
    param.requires_grad = False


train_dataset_raw   = datasets.ImageFolder(train_dir, transform=None)
val_dataset_raw   = datasets.ImageFolder(val_dir, transform=None)

train_dataset = FoveatedUpletDataset(
                base_dataset      = train_dataset_raw,
                zoom=zoom,
                std=std,
                n_uplet=n_uplet,
                start_center=False, 
                output_size       = 128
            )

val_dataset = FoveatedUpletDataset(
                base_dataset      = val_dataset_raw,
                zoom=zoom,
                std=std,
                n_uplet=n_uplet,
                start_center=False, 
                output_size       = 128
            )


#train_loader = make_dataloader(train_dataset_raw, shiftzoom_transform, fovea_transform, batch_size=batch_size) #, limit=500)

#val_loader = make_dataloader(train_dataset_raw, shiftzoom_transform, fovea_transform, batch_size=batch_size) #, limit=batch_size*5)

train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size  = batch_size,
                shuffle     = True,
                num_workers = num_workers,
                #pin_memory  = True,
            )

val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size  = batch_size,
                shuffle     = True,
                num_workers = num_workers,
                #pin_memory  = True,
            )



# Définir la tête linéaire
linear_head = nn.Linear(model_orig.embed_dim, 1000)
linear_head.to(device)
linear_head.train()

os.makedirs(save_dir, exist_ok=True)

# Optimiseur
#optimizer = torch.optim.SGD(linear_head.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.SGD(
    linear_head.parameters(),
    lr=0.01,              # Facebook utilise 0.01 avec batch_size=256
    momentum=0.9,
    weight_decay=1e-4,    # régularisation importante
    nesterov=True,
)

scaler = torch.cuda.amp.GradScaler()

criterion = nn.CrossEntropyLoss()

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
        "loss": [], "classif": []} #, "train_loss_3": []}

os.makedirs(save_dir, exist_ok=True)

for epoch in range(train_epochs):  # 20-30 époques suffisent

    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs}")

    for batch_idx, (mosaics, sxs, sys_, zooms, labels) in enumerate(pbar):

        mosaics = mosaics.to(device)          # (B, V, 3, H, W)
        sxs     = sxs.to(device)             # (B, V)
        sys_    = sys_.to(device)
        zooms   = zooms.to(device)
        labels   = labels.to(device)
        #images = [img.to(device, non_blocking=True) for img in images]
        #labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = model(mosaics)  # Extraction des features
            output = linear_head(features[:,0])
            loss = criterion(output, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:

            print(f"Epoch {epoch+1:03d} | simple loss = {total_loss / log_interval:.4f}")

            history["epoch"].append(epoch + 1)
            history["batch"].append(batch_idx + 1)
            history["loss"].append(total_loss / log_interval)

            total_loss = 0

            total = 0
            correct = 0.0
            val_iter = iter(val_loader)

            with torch.no_grad():
                for _ in range(5):
                    mosaics, sxs, sys_, zooms, labels = next(val_iter)

                    mosaics = mosaics.to(device)          # (B, V, 3, H, W)
                    sxs     = sxs.to(device)             # (B, V)
                    sys_    = sys_.to(device)
                    zooms   = zooms.to(device)
                    labels   = labels.to(device)
                    #images = [img.to(device, non_blocking=True) for img in images]
                    #labels = labels.to(device, non_blocking=True)

                    features = model(mosaics)
                    output = linear_head(features[:,0,...])
                    loss = criterion(output, labels)

                    preds = output.argmax(dim=1)

                    correct += (preds == labels).sum().item()

                    total += labels.size(0)

            print(f"Top-1 accuracy: {100 * correct / total:.2f}%")

            history["classif"].append(100 * correct / total)

            torch.save({
                    "epoch": epoch,
                    "history": history,
                    "classifier": linear_head.state_dict(),
                },  os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
            
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, "training_log_teach_{epoch_teacher}.csv"), index=False)        

    scheduler.step()

# Démonter le dossier à la fin
subprocess.run(["fusermount", "-u", mount_point], check=True)
 



