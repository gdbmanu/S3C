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
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

from pathlib import Path


from s3c.models.foveated_vit import FoveatedMultiViT, build_foveated_pos_embed
from s3c.models.heads import FovealSetTransformer
from s3c.data.datasets import FoveatedUpletDataset, FoveatedGridDataset, make_dataloader
from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform

import timm

from PIL import Image

from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 14 #56
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

n_uplet = 30
grid=True
n_grid = 11


# Monter le dossier distant
local=True
if local == False:
    mount_point = os.path.expanduser("~/imagenet_mount")

    try:
        os.makedirs(mount_point, exist_ok=True)
        subprocess.run(["sshfs", "dauce.e@brain-lid-004:data/Imagenet_full", mount_point, "-o", "reconnect"], check=True)
    except:
        pass

    train_dir = os.path.join(mount_point, "train")
    val_dir = os.path.join(mount_point, "val")
else:
    train_dir = "~/data/Imagenet_full/train"   # Imagenet Validation set
    val_dir = "~/data/Imagenet_full/val"   # Imagenet Validation set

load_dir = "../checkpoints/checkpoints_EMA_Xattn_260416"

save_dir = "data/Imagenet_grid_Z"

epoch_teacher = 20

zoom = 1.5

std = 0.5 / zoom

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

model = FoveatedMultiViT(model_orig, norm=False)

# --- Chargement dans le modèle ---
missing, unexpected = model.model.load_state_dict(state_dict, strict=False)

print("➡️ Poids chargés (teacher).")
print("❗ Paramètres manquants :", missing)
print("⚠️ Paramètres inattendus :", unexpected)


model.model.norm = nn.Identity() # pas de normalisation en sortie !
model.to(device)

# 2. Geler le backbone
for param in model.parameters():
    param.requires_grad = False

train_dataset_raw   = datasets.ImageFolder(train_dir, transform=None)
val_dataset_raw   = datasets.ImageFolder(val_dir, transform=None)

# ── fonction principale ───────────────────────────────────────────────────
@torch.no_grad()
def generate_embeddings(model, split, dataset_raw, output_dir,
                        n_sac, zoom, std,
                        batch_size, num_workers, device):
    """
    Génère et sauvegarde les embeddings pour un split (train ou val).
    """
    print(f"\n── Génération embeddings — {split} ──────────────────────────")
    
    if not grid:
        dataset = FoveatedUpletDataset(
            base_folder=dataset_raw, # ImageFolder
            zoom=zoom,
            std=std,
            start_center=False, 
            n_uplet=n_sac,
            path=True
        )
    else:
        dataset = FoveatedGridDataset(
            base_folder=dataset_raw, # ImageFolder
            zoom=zoom,
            n_grid=n_grid,
            path=True
        )

    from torch.utils.data import Subset

    '''# Calculer la taille des 10% finaux
    dataset_size = len(dataset)
    last_7_percent_size = int(0.07 * dataset_size)
    start_idx = dataset_size - last_7_percent_size

    # Créer un sous-ensemble avec les derniers 10%
    subset_indices = range(start_idx, dataset_size)
    last_7_percent_dataset = Subset(dataset, subset_indices)

    # Créer le DataLoader sur ce sous-ensemble
    loader = DataLoader(
        last_7_percent_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )'''

     
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
    )

    model.eval()
    out_root = Path(output_dir) / split
    skipped  = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() \
                else torch.float16

    for batch_idx, (imgs, xs, ys, zooms, labels, rel_paths) in enumerate(tqdm(loader)):

        skip = False
        for i, rel_path in enumerate(rel_paths):
            out_path = out_root / rel_path
            out_path = out_path.with_suffix(".pt")    # remplace .JPEG par .pt

            # skip si déjà généré (reprise après interruption)
            if out_path.exists():
                skipped += 1
                print(f'{out_path} skip')
                skip = True
                continue
        if skip:
            continue

        # imgs : (B, n_sac, 3, H, W)
        imgs = imgs.to(device)

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            with torch.inference_mode():
                z = model(imgs)    # (B, n_sac, D)

        # sauvegarder chaque embedding dans un dossier séparé
        for i, rel_path in enumerate(rel_paths):
            out_path = out_root / rel_path
            out_path = out_path.with_suffix(".pt")    # remplace .JPEG par .pt

            # skip si déjà généré (reprise après interruption)
            '''if out_path.exists():
                skipped += 1
                print(f'{out_path} skip')
                continue'''

            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Sauvegarder z[i], xs[i], et ys[i] dans un dictionnaire
            data_to_save = {
                "zs": z[i].half().cpu(),
                "xs": xs[i].cpu(),  # 
                "ys": ys[i].cpu(),  # 
            }

            torch.save(data_to_save, out_path)

    print(f"  Terminé — {skipped} fichiers déjà existants ignorés.")



 # ── lancement ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # charger modèle

    for split, dataset_raw in [("train", train_dataset_raw),
                               ("val",   val_dataset_raw)]:
        generate_embeddings(
            model           = model,       # FoveatedMultiViT déjà chargé
            split           = split,
            dataset_raw     = dataset_raw,
            output_dir      = save_dir,
            n_sac           = n_uplet,
            zoom            = zoom,
            std             = std,
            batch_size      = batch_size,
            num_workers     = num_workers,
            device          = device,
        )


