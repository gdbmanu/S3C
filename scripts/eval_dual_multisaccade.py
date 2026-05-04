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
from s3c.models.heads import DualPredictor
from s3c.data.datasets import make_dataloader

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
local=False

if not local:
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

epoch_teacher = 20

dual_head_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_dual"
#linear_head_dir = "./checkpoints_260414_base_1_view"


dual_epoch = 10

train_epochs = 20

n_views = 10

zoom = 1.5

save_dir = "../checkpoints/checkpoints_260427_EMA_Xattn_dual"


std = 0.5 / zoom

model_name = "vit_base_patch8_224.dino"
model_orig = timm.create_model(model_name, pretrained=True)

model_orig.patch_embed.strict_img_size = False
norm_ref = copy.deepcopy(model_orig.norm)

build_foveated_pos_embed(model_orig,
                        base_img_size=224,
                        patch_size=8,
                        mosaic_sub_patch_grid=8,   # 8 patches par sous-image (64px/8)
                        scales=(1.0, 0.5, 0.25, 0.125))

model = FoveatedMultiViT(model_orig, norm=True)

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


model.model.norm = norm_ref #= !!IMPORTANT : AVEC NORME !! (nn.Identity() = pas de normalisation en sortie)
model.to(device)
model.eval()

# 2. Geler le backbone
for param in model.parameters():
    param.requires_grad = False

# Définir la tête 

dual_predictor = DualPredictor(model_orig.embed_dim, 256)
dual_predictor.to(device)
dual_predictor.train()


checkpoint_path = os.path.join(dual_head_dir, f"checkpoint_epoch{dual_epoch}.pt")  # exemple
checkpoint = torch.load(checkpoint_path, map_location="cpu")
# Vérifie les clés disponibles
print("Clés du checkpoint :", checkpoint.keys())
state_dict = checkpoint["dual_predictor"]

# --- Chargement dans le modèle ---
missing, unexpected = dual_predictor.load_state_dict(state_dict, strict=False)

print("➡️ Poids chargés (dual head).")
print("❗ Paramètres manquants :", missing)
print("⚠️ Paramètres inattendus :", unexpected)

dual_predictor.to(device)
dual_predictor.eval()


# %%
from tqdm import tqdm

os.makedirs(save_dir, exist_ok=True)

val_dataset_raw   = datasets.ImageFolder(val_dir, transform=None)

with torch.no_grad():

    val_loader = make_dataloader(val_dataset_raw, zoom, std, n_views, start_center=True, batch_size=batch_size, num_workers=num_workers, limit=None)


    total = 0
    correct = np.zeros(n_views)

    pbar = tqdm(val_loader)


    for views, sxs, sys_, zooms, labels in pbar:
        views = views.to(device)          # (B, V, 3, H, W)
        labels   = labels.to(device)

        features = model(views) # (B, V, D) 

        for i in range(1, n_views):
            pred_shift, output  = dual_predictor(features[:,0,:], features[:,i,:])
            if i == 1:
                sum_output = output
            else:
                sum_output += output

            preds = sum_output.argmax(dim=1)

            correct[i] += (preds == labels).sum().item()

        total += labels.size(0)

        history = {"n_saccades": [], 
            "classif": []}

        for i in range(n_views):
            history["n_saccades"].append(i + 1)
            print(f"Top-1 accuracy: {100 * correct[i] / total:.2f}%")
            history["classif"].append(100 * correct[i] / total)

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(save_dir, "training_log_multi.csv"), index=False)        

# Démonter le dossier à la fin
subprocess.run(["fusermount", "-u", mount_point], check=True)
 

