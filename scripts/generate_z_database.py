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
from s3c.data.datasets import FoveatedUpletDataset, make_dataloader
from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform

import timm

from PIL import Image

from tqdm import tqdm


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 128
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

n_uplet = 30


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

save_dir = "~/data/Imagenet_Z"

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
model.eval()

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
    
    dataset = FoveatedUpletDataset(
        root=dataset_raw, # ImageFolder
        zoom=zoom,
        std=std,
        start_center=False, 
        n_uplet=n_sac,
        path=True
    )
     
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
    
    pbar = tqdm(loader, desc=f"Batch {batch_idx+1}/{len(loader)/batch_size}")

    for batch_idx, (imgs, xs, ys, zooms, labels, rel_paths) in enumerate(pbar):
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
            if out_path.exists():
                skipped += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Sauvegarder z[i], xs[i], et ys[i] dans un dictionnaire
            data_to_save = {
                "zs": z[i].half().cpu(),
                "xs": xs[i].cpu(),  # 
                "ys": ys[i].cpu(),  # 
            }

            torch.save(data_to_save, out_path)

    print(f"  Terminé — {skipped} fichiers déjà existants ignorés.")


# ── dataset pour charger les embeddings pré-calculés ─────────────────────
class ImageNetZDataset(Dataset):
    """
    Charge les embeddings pré-calculés depuis imagenet_z/.
    Retourne (z, label) avec z de shape (n_sac, D).
    """
    def __init__(self, root):
        self.root    = Path(root)
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir()
                                if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            label = self.class_to_idx[cls]
            for pt in (self.root / cls).glob("*.pt"):
                self.samples.append((pt, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        dict = torch.load(path, map_location="cpu") #.float()  # (n_sac, D)
        zs = dict["zs"].float()
        xs = dict["xs"].float()
        ys = dict["ys"].float()
        return zs, xs, ys, label

    def __len__(self):
        return len(self.samples)


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

    # test du dataset de chargement
    zs_train, xs_train, ys_train = ImageNetZDataset(f"{save_dir}/train")
    zs_val, xs_val, ys_val = ImageNetZDataset(f"{save_dir}/val")
    print(f"\nTrain : {len(zs_train)} embeddings")
    print(f"Val   : {len(zs_val)} embeddings")

    z, label = zs_train[0]
    print(f"Shape embedding : {z.shape}")   # (n_uplet, 768)
    print(f"Label           : {label}")      


