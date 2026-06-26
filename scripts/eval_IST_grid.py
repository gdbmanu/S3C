import os
import pandas as pd
import numpy as np

import copy

import subprocess

from urllib.request import urlopen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets

from s3c.models.foveated_vit import build_foveated_pos_embed, FoveatedMultiViT
from s3c.data.datasets import make_dataloader
from s3c.models.heads import IterativeSeedTransformer, AttentionPooling #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset

import timm

from tqdm import tqdm

from datetime import datetime



## Datasets

'''mount_point = os.path.expanduser("~/imagenet_full")
try:
    os.makedirs(mount_point, exist_ok=True)
    subprocess.run(["sshfs", "dauce.e@brain-lid-004:Recherche/scripts/S3C/scripts/data/Imagenet_full", mount_point, "-o", "reconnect"], check=True)
except:
    pass
imgnet_train_dir = os.path.join(mount_point, "train")
imgnet_val_dir = os.path.join(mount_point, "val")'''




# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

weight_dirs = {"orig" : "260613_IST3+ABMIL_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_TEST_CROSS_ORIG_s3_t5_space",
           "teacher": "260617_IST3+ABMIL_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a1e-06_TEST_CROSS_APOS2_s3_t5_space", #"260614_IST3+ABMIL_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_alph1e-06_TEST_CROSS_s3_t5_space_(*)",
           "grid" : "260613_IST3+ABMIL_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_TEST_CROSS_GRID_s3_t7_space"}

history = {}

for weight_dir_key in weight_dirs:

    batch_size = 256

    load_dir = "../checkpoints/" + weight_dirs[weight_dir_key]

    n_views = 10

    zoom = 1.5
    std = 0.5 / zoom

    ### Load weights

    epoch_ist = 30
    k = 3
    n_heads = 12
    n_sab = 2
    self_att = False


    checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_ist}.pt")  # exemple
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Clés du checkpoint :", checkpoint.keys())

    ist_transformer = IterativeSeedTransformer(input_dim=embed_dim, d_model=embed_dim,
                    n_heads=n_heads, n_seeds=k, n_blocks=n_sab, self_att=self_att)

    for param in ist_transformer.parameters():
        param.requires_grad = False

    if "ist_transformer" not in checkpoint:
        raise KeyError(f"Aucune clé 'ist_transformer' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["ist_transformer"]
    missing, unexpected = ist_transformer.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (ist_transformer).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    ist_transformer.to(device)
    ist_transformer.eval()

    linear_head = nn.Sequential(
                    nn.Unflatten(1, (k, embed_dim)),          # (B, k*d) → (B, k, d)
                    nn.LayerNorm(embed_dim),                  # norm par seed ✓
                    nn.Flatten(1),    
                    nn.Linear(k * embed_dim, 1000),
                )

    if "linear_head" not in checkpoint:
        raise KeyError(f"Aucune clé 'linear_head' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["linear_head"]
    missing, unexpected = linear_head.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (linear_head).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    linear_head.to(device)
    linear_head.eval()

    ## save_dir
    save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_IST{k}_grid_EVAL"

    os.makedirs(save_dir, exist_ok=True)

    if weight_dir_key == "orig":
        val_dir = "data/Imagenet_grid_orig_Z/val"
    else:
        val_dir = "data/Imagenet_grid_Z/val"   # Imagenet Validation set


    val_dataset   = ImageNetZDataset(val_dir) 

    val_loader = DataLoader(
            val_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
        )

    with torch.no_grad():

        total = 0
        correct = np.zeros(n_views)

        pbar = tqdm(val_loader)

        total = 0
        correct = 0.0
        for features, sxs, sys_, labels  in pbar:
            features = features.to(device)          # (B, V, 3, H, W)
            sxs     = sxs.to(device)             # (B, V)
            sys_    = sys_.to(device)
            labels   = labels.to(device)

            batch_size, n = sxs.shape

            z_seeds = ist_transformer(features)
            output_t_head = linear_head(z_seeds.view(batch_size, k * embed_dim).detach())

            '''for i in range(n_views):
                features = model(mosaics[:, i])  # Extraction des features
                if i == 0:
                    output = linear_head(features[:,0])
                else:
                    output += linear_head(features[:,0])
            '''
            preds = output_t_head.argmax(dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            print(f"{weight_dir_key} Batch accuracy: {100 * (preds == labels).sum().item() / batch_size:.2f}%")

        '''for i in range(n_views):
            history["n_saccades"].append(i + 1)
            print(f"Top-1 accuracy: {100 * correct[i] / total:.2f}%")
            history["classif"].append(100 * correct[i] / total)'''
        
    history[f"{weight_dir_key} test_classif"] = [100 * correct / total]

    print(f"{weight_dir_key} Global accuracy: {100 * correct / total:.2f}%")

df = pd.DataFrame(history)
df.to_csv(os.path.join(save_dir, "training_log_grid_ref.csv"), index=False)        

# Démonter le dossier à la fin
# subprocess.run(["fusermount", "-u", mount_point], check=True)
 

