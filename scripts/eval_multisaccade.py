import os
import pandas as pd
import numpy as np

import copy

import subprocess

from urllib.request import urlopen

import torch
import torch.nn as nn

from torchvision import datasets

from s3c.models.foveated_vit import build_foveated_pos_embed, FoveatedMultiViT
from s3c.data.datasets import ShiftZoomUplet, FoveatedUpletDataset

import timm


from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 256
num_workers = 4
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

# Monter le dossier distant

mount_point = os.path.expanduser("~/imagenet_mount")

try:
    os.makedirs(mount_point, exist_ok=True)
    subprocess.run(["sshfs", "dauce.e@brain-lid-004:data/Imagenet_full", mount_point, "-o", "reconnect"], check=True)
except:
    pass

# Ton code ici, en utilisant mount_point
train_dir = os.path.join(mount_point, "train")
val_dir = os.path.join(mount_point, "val")


# train_dir = "~/data/Imagenet_full/train"   # Imagenet Validation set
# val_dir = "~/data/Imagenet_full/val"   # Imagenet Validation set

load_dir = "../checkpoints/checkpoints_EMA_Xattn_260321"

epoch_teacher = 20

linear_head_dir = "../checkpoints/checkpoints_260414_EMA_Xattn_1_view"
#linear_head_dir = "./checkpoints_260414_base_1_view"


linear_epoch = 20

train_epochs = 20

n_views = 10

zoom = 1.5

save_dir = "../checkpoints/checkpoints_260414_EMA_Xattn_1_view"
#save_dir = "./checkpoints_260319_base_1_view"

zoom = 1.5

std = 0.5 / zoom

# Charger un ViT Tiny via timm

#img = Image.open(urlopen(
#    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
#))

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


checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_teacher}.pt")  # exemple
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# Vérifie les clés disponibles
print("Clés du checkpoint :", checkpoint.keys())
# --- Récupération des poids du teacher ---
if "teacher" not in checkpoint:
    raise KeyError(f"Aucune clé 'teacher' trouvée dans {checkpoint_path}")

state_dict = checkpoint["teacher"]

model = FoveatedMultiViT(model_orig)

if True:
    # --- Chargement dans le modèle ---
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

    print("➡️ Poids chargés (teacher).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)


model.model.norm = norm_ref #= !!IMPORTANT : AVEC NORME !! (nn.Identity() = pas de normalisation en sortie)
model.to(device)
model.eval()

# 2. Geler le backbone
for param in model.parameters():
    param.requires_grad = False

# Définir la tête linéaire
linear_head = nn.Linear(model_orig.embed_dim, 1000)

checkpoint_path = os.path.join(linear_head_dir, f"checkpoint_epoch{linear_epoch}.pt")  # exemple
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# Vérifie les clés disponibles
print("Clés du checkpoint :", checkpoint.keys())
state_dict = checkpoint["classifier"]

# --- Chargement dans le modèle ---
missing, unexpected = linear_head.load_state_dict(state_dict, strict=False)

print("➡️ Poids chargés (linear head).")
print("❗ Paramètres manquants :", missing)
print("⚠️ Paramètres inattendus :", unexpected)

linear_head.to(device)
linear_head.eval()

#c = input('continuer?')

# %%
from tqdm import tqdm


os.makedirs(save_dir, exist_ok=True)

history = {"n_saccades": [], 
            "classif": []}

os.makedirs(save_dir, exist_ok=True)

val_dataset_raw   = datasets.ImageFolder(val_dir, transform=None)

with torch.no_grad():

    uplet_tf = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=n_views)


    val_dataset = FoveatedUpletDataset(
                    root              = val_dataset_raw,
                    shift_zoom_uplet  = uplet_tf,
                    output_size       = 128,
                    resize            = 512,
                    crop              = 512,
                )


    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size  = batch_size,
                    shuffle     = True,
                    num_workers = num_workers,
                    pin_memory  = True,
                )

    total = 0
    correct = np.zeros(n_views)

    pbar = tqdm(val_loader)


    for mosaics, sxs, sys_, zooms, labels in pbar:
        mosaics = mosaics.to(device)          # (B, V, 3, H, W)
        sxs     = sxs.to(device)             # (B, V)
        sys_    = sys_.to(device)
        zooms   = zooms.to(device)
        labels   = labels.to(device)

        for i in range(n_views):
            features = model(mosaics[:, i])  # Extraction des features
            if i == 0:
                output = linear_head(features[:,0])
            else:
                output += linear_head(features[:,0])

            preds = output.argmax(dim=1)

            correct[i] += (preds == labels).sum().item()

        total += labels.size(0)

    for i in range(n_views):
        history["n_saccades"].append(i + 1)
        print(f"Top-1 accuracy: {100 * correct[i] / total:.2f}%")
        history["classif"].append(100 * correct[i] / total)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "training_log_multi.csv"), index=False)        

# Démonter le dossier à la fin
subprocess.run(["fusermount", "-u", mount_point], check=True)
 

