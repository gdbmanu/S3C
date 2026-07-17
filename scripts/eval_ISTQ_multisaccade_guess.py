import os
import pandas as pd
import numpy as np

import copy

import subprocess

from urllib.request import urlopen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets
from torchvision.models import ResNet50_Weights
from torchvision.datasets import ImageFolder

from s3c.models.foveated_vit import build_foveated_pos_embed, FoveatedMultiViT
from s3c.data.datasets import make_dataloader
from s3c.models.heads import IterativeSeedTransformerwithQuery, AttentionPooling, ABMILPosPredictor #FovealSetTransformer
from s3c.data.datasets import ImageNetZDataset
from s3c.utils.training import get_parent_synset


import timm

from tqdm import tqdm

from datetime import datetime


# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim =  768

weight_dirs = {#"orig" : "260702_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH0.8_APOS2_ORIG_s3_t5_space",
           #"teacher": "260701_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH0.8_APOS2_s3_t5_space", 
           #"teacher": "260707_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH0.5_SYNSET_APOS2_s3_t5_space",
           #"teacher": "260708_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH0.5_SYNSET6_APOS2_s3_t5_space",
           #"teacher": "260630_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_RES_DETACH_SMOOTH_APOS2_s3_t5_space",
           "grid": "260708_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_FINE_GRID_RES_DETACH_SMOOTH0.5_SYNSET_APOS2_s6_t15_space" #"260702_ISTQ_3_semi_z_lam0.05_mu_1_sab2_LeJ_SUP_a3e-06_TEST_CROSS_GRID_RES_DETACH_SMOOTH0.8_APOS2_s3_t7_space"
           }

history = {}

n_grid = 11

n_views = 4

verb=False

use_synset_embeddings = True
oracle = False
pos_oracle = False
bootstrap = False and not oracle
random_guess = True and not pos_oracle

rand_init = True
n_warmup = 3

classes = ResNet50_Weights.DEFAULT.meta['categories']



    

def pos_to_grid_idx(pos_pred, used_mask, n_grid=11,
                    grid_min=-0.75, grid_max=0.75):
    """
    Retourne aussi les coordonnées (x, y) de la position choisie
    pour permettre la vérification.
    """
    B        = pos_pred.shape[0]
    device   = pos_pred.device

    grid_vals = torch.linspace(grid_min, grid_max, n_grid, device=device)
    #gy, gx    = torch.meshgrid(grid_vals, grid_vals, indexing='ij')
    #grid_xy   = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (n_total, 2)

    gx, gy  = torch.meshgrid(grid_vals, grid_vals, indexing='xy')
    grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (n_total, 2)

    dist = (pos_pred.unsqueeze(1) - grid_xy.unsqueeze(0)).pow(2).sum(dim=-1)
    dist = dist.masked_fill(used_mask, float('inf'))

    idx_next = dist.argmin(dim=-1)   # (B,)

    # Coordonnées réelles de la position choisie
    xy_chosen = grid_xy[idx_next]    # (B, 2)

    return idx_next, xy_chosen


def evaluate_multisaccade(val_loader, ist_transformer, linear_head, pos_predictor,
                          n_views, n_grid=11, oracle=False,
                          device='cuda', verb=False):
    n_total_positions = n_grid * n_grid
    center_idx        = (n_grid * n_grid) // 2   # position 60 pour grille 11x11

    total   = 0
    correct = np.zeros(n_views)

    with torch.no_grad():
        for i, (features, sxs, sys_, labels) in enumerate(tqdm(val_loader)):
            features = features.to(device)   # (B, n_grid², 768)
            labels   = labels.to(device)
            sxs = sxs.to(device)
            sys_ = sys_.to(device)

            batch_size = labels.size(0)

            in_labels = labels if oracle else None
            in_labels = label_to_synset_tensor[in_labels] if oracle and use_synset_embeddings else in_labels 

            perms = torch.stack([torch.randperm(n_saccades_max) for _ in range(batch_size)])


            # ── Initialisation ────────────────────────────────────────
            # Masque des positions utilisées : (B, n_grid²)
            used_mask = torch.zeros(
                batch_size, n_total_positions, dtype=torch.bool, device=device
            )

            if rand_init: # random initial saccade
                center_idx = perms[:,0]

            #print(center_idx.shape)    

            # Première saccade : toujours le centre (position 60)
            used_mask[torch.arange(batch_size, device=device), center_idx] = True

            # Vues accumulées : on commence au centre
            views_acc = features[torch.arange(batch_size, device=device), center_idx, :].unsqueeze(1)  # (B, 1, 768)

            # ── Boucle de saccades ────────────────────────────────────
            for num_view in range(n_views):

                # Forward
                z_seeds = ist_transformer(
                    views_acc, in_labels, None
                )

                if pos_oracle:
                    z_seeds_oracle = ist_transformer(
                        views_acc, label_to_synset_tensor[labels] if use_synset_embeddings else labels, None
                        ) 
                    pos_in = z_seeds_oracle[:, k+1, :]           
                else:
                    pos_in = z_seeds[:, k+1, :]   

                # Classification
                logits = linear_head(z_seeds[:, k, :])

                # z_test = ist_transformer(
                #     views_acc[:,-1,:].unsqueeze(1), in_labels, None
                # )

                # pos_out, _ = pos_predictor(z_test[:,:k,:], z_test[:, k+1, :])


                pos_out, _ = pos_predictor(z_seeds[:,:k,:], pos_in)
                #print(pos_out[0,:].cpu().detach().numpy())
                preds  = logits.argmax(dim=1)
                correct[num_view] += (preds == labels).sum().item()

                if bootstrap :
                    if use_synset_embeddings:
                        in_labels = label_to_synset_tensor[preds]   # (B,) — conversion immédiate
                        #topk_vals, topk_indices = torch.topk(logits, k=3, dim=-1)
                        #in_labels = label_to_synset_tensor[topk_indices[:, num_view % 3]]
                    # Top-k : à la saccade i, explorer la direction du top-(i+1)
                    else:
                        topk_vals, topk_indices = torch.topk(logits, k=n_views, dim=-1)
                        in_labels = topk_indices[:, num_view % 3]   # (B,) — top2, top3...
                else:
                    in_labels = in_labels

                # Choisir la prochaine saccade (sauf à la dernière itération)
                if num_view < n_views - 1:
                    #idx_next = pos_to_grid_idx(
                    #    pos_out, used_mask, n_grid=n_grid
                    #)                                      # (B,)
                    #print(pos_out[0,:].cpu().detach().numpy(), idx_next[0].item())

                    idx_next, xy_chosen = pos_to_grid_idx(
                        pos_out, used_mask, n_grid=n_grid
                    )   # (B,), (B, 2)

                    if num_view==1:
                        if use_synset_embeddings:
                            if bootstrap:
                                print(f"true label :  {synset_names[label_to_synset_tensor[labels[0]].item()]}-{classes[labels[0].item()]} -- label guess : {synset_names[in_labels[0].item()]}-{classes[preds[0].item()]} -- pos predit : {pos_out[0,0].item():.2f},{pos_out[0,1].item():.2f} -- indice: {idx_next[0].item()} -- grille : {xy_chosen[0,0].item():.2f},{xy_chosen[0,1].item():.2f}")
                            else:
                                print(f"true label :  {synset_names[label_to_synset_tensor[labels[0]].item()]}-{classes[labels[0].item()]} -- label guess : -{classes[preds[0].item()]} -- pos predit : {pos_out[0,0].item():.2f},{pos_out[0,1].item():.2f} -- indice: {idx_next[0].item()} -- grille : {xy_chosen[0,0].item():.2f},{xy_chosen[0,1].item():.2f}")
                        else:
                            print(f"true label :  {classes[labels[0].item()]} -- label guess : {classes[in_labels[0].item()]} -- pos predit : {pos_out[0,0].item():.2f},{pos_out[0,1].item():.2f} -- indice: {idx_next[0].item()} -- grille : {xy_chosen[0,0].item():.2f},{xy_chosen[0,1].item():.2f}")

                    if random_guess or num_view < n_warmup:
                        idx_next = perms[:,num_view+1]

                    if verb:
                        # ── Vérification pos_pred vs grille vs sxs/sys_ ──────────────────
                        # Récupérer les vraies coordonnées de la position choisie
                        # depuis sxs/sys_ (shape : (B, n_grid²))
                        sx_real = sxs[torch.arange(batch_size), idx_next]   # (B,)
                        sy_real = sys_[torch.arange(batch_size), idx_next]  # (B,)

                        # Erreur entre coordonnées grille et coordonnées dataset
                        err_x = (xy_chosen[:, 0] - sx_real).abs().mean().item()
                        err_y = (xy_chosen[:, 1] - sy_real).abs().mean().item()

                        # Erreur entre pos_pred brute et vraie position
                        err_pred_x = (pos_out[:, 0] - sx_real).abs().mean().item()
                        err_pred_y = (pos_out[:, 1] - sy_real).abs().mean().item()

                        if num_view == 0:   # afficher seulement à la première saccade
                            print(f"  Grille→dataset  : err_x={err_x:.4f}  err_y={err_y:.4f}")
                            print(f"  Pred→dataset    : err_x={err_pred_x:.4f}  err_y={err_pred_y:.4f}")

                        # Assertion optionnelle — erreur grille doit être < pas de grille
                        step = 1.5 / (n_grid - 1)   # pas de grille ≈ 0.15 pour n_grid=11
                        assert err_x < step / 2 and err_y < step / 2, \
                            f"Désalignement grille/dataset : err_x={err_x:.4f}, err_y={err_y:.4f}"

                    # Marquer comme utilisée
                    used_mask[torch.arange(batch_size), idx_next] = True

                    # Extraire la feature à la position choisie
                    z_new = features[
                        torch.arange(batch_size, device=device),
                        idx_next, :
                    ].unsqueeze(1)                         # (B, 1, 768)

                    # Accumuler les vues
                    views_acc = torch.cat([views_acc, z_new], dim=1)
                
            total += batch_size

            if i % 20 == 19:
                for num_view in range(n_views):
                    print(f"{weight_dir_key} {num_view} running accuracy: {100 * correct[num_view] / total:.2f}%")      

    accuracy_per_view = 100 * correct / total
    return accuracy_per_view

# Test rapide hors boucle
grid_vals = torch.linspace(-0.75, 0.75, 11)
gy, gx = torch.meshgrid(grid_vals, grid_vals, indexing='ij')
grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=1)

print(f"idx=0  : {grid_xy[0]}")    # doit être (-0.75, -0.75)
print(f"idx=60 : {grid_xy[60]}")   # doit être ( 0.00,  0.00)
print(f"idx=120: {grid_xy[120]}")  # doit être ( 0.75,  0.75)


for weight_dir_key in weight_dirs:

    batch_size = 256

    load_dir = "../checkpoints/" + weight_dirs[weight_dir_key]

    
    
    zoom = 1.5
    std = 0.5 / zoom

    ### Load weights

    if weight_dir_key == "grid":
        epoch_ist = 100
        n_synsets = 4
    else:
        epoch_ist = 30
        n_synsets = 6


    if use_synset_embeddings:
        data = torch.load(f'imagenet_synset_{n_synsets}_embeddings.pt')
        label_to_synset_tensor = data['label_to_synset'].to(device)
        n_synsets              = data['n_synsets']
        synset_names           = data['synset_names']
        synset_embeddings      = data['embeddings'].to(device)    

    k = 3
    n_heads = 12
    n_sab = 2
    residual = True
    l_emb_detach = True
    label_smoothing = 0.5


    checkpoint_path = os.path.join(load_dir, f"checkpoint_epoch{epoch_ist}.pt")  # exemple
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Clés du checkpoint :", checkpoint.keys())

    if use_synset_embeddings:
        ist_transformer = IterativeSeedTransformerwithQuery(n_heads=n_heads, 
                                                            n_seeds=k, 
                                                            n_blocks=n_sab, 
                                                            pretrained_embeddings=None,
                                                            n_classes=n_synsets,
                                                            residual=residual, 
                                                            l_emb_detach=l_emb_detach, 
                                                            label_smoothing=label_smoothing)
    else:
        ist_transformer = IterativeSeedTransformerwithQuery(n_heads=n_heads, 
                                                            n_seeds=k, 
                                                            n_blocks=n_sab, 
                                                            pretrained_embeddings=None,
                                                            residual=residual, 
                                                            l_emb_detach=l_emb_detach, 
                                                            label_smoothing=label_smoothing)


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
                    nn.LayerNorm(embed_dim),                  
                    nn.Linear(embed_dim, 1000),
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

    pos_predictor = ABMILPosPredictor(embed_dim, k)

    if "pos_predictor" not in checkpoint:
        raise KeyError(f"Aucune clé 'pos_predictor' trouvée dans {checkpoint_path}")
    state_dict = checkpoint["pos_predictor"]
    missing, unexpected = pos_predictor.load_state_dict(state_dict, strict=False)
    print("➡️ Poids chargés (pos_predictor).")
    print("❗ Paramètres manquants :", missing)
    print("⚠️ Paramètres inattendus :", unexpected)

    pos_predictor.to(device)
    pos_predictor.eval()

    ## save_dir
    save_dir = f"../checkpoints/{datetime.now().strftime('%y%m%d')}_ISTQ_{k}_multi_EVAL"

    os.makedirs(save_dir, exist_ok=True)

    if weight_dir_key == "orig":
        val_dir = "data/Imagenet_grid_orig_Z/val"
        n_saccades_max = 121
    else:
        val_dir = "data/Imagenet_grid_Z/val"   # Imagenet Validation set
        n_saccades_max = 121


    val_dataset   = ImageNetZDataset(val_dir) 

    val_loader = DataLoader(
            val_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
        )

    accuracy_per_view = evaluate_multisaccade(val_loader, ist_transformer, linear_head, pos_predictor,
                          n_views, n_grid=11, oracle=oracle, verb=verb,
                          device='cuda')

            
    if weight_dir_key == "orig":
        history["n_saccades"] = []
    history[f"{weight_dir_key} classif"] = []
    for num_view in range(n_views):
        if weight_dir_key == "orig":
            history["n_saccades"].append(num_view + 1)
        print(f"{weight_dir_key} {num_view} accuracy: {accuracy_per_view[num_view]:.2f}%")
        history[f"{weight_dir_key} classif"].append(accuracy_per_view[num_view])
        
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, f"training_log_multi_guess{'_BOOT' if bootstrap else ''}{{f'_WARMUP{n_warmup}' if n_warmup>0 else ''}}{'_SYNSET' if use_synset_embeddings else ''}{'_ORACLE' if oracle else ''}{'_POS_ORACLE' if pos_oracle else ''}{'_RND' if random_guess else ''}.csv"), index=False)        

# Démonter le dossier à la fin
# subprocess.run(["fusermount", "-u", mount_point], check=True)
 

