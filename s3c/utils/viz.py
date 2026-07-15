import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np

import torch
import torch.nn.functional as F

from s3c.data.datasets import ShiftZoomUplet

def foveate_mosaic(img_mosaic, attn=False):
    """
    img_mosaic : Tensor [B, C, 128, 128]
    Retourne une image fovéée [B, C, 512, 512]
    Organisation mosaïque :
      [ global | intermédiaire ]
      [ péri-fovéal | fovéal ]
    """
    B, C, H, W = img_mosaic.shape
    #assert H == 128 and W == 128, "Entrée attendue : 128x128 mosaïque"

    # --- 1. Séparer les imagettes 64x64 ---
    h2, w2 = H // 2, W // 2
    global_img       = img_mosaic[:, :, 0:h2, 0:w2]
    intermediate_img = img_mosaic[:, :, 0:h2, w2:W]
    perifoveal_img   = img_mosaic[:, :, h2:H, 0:w2]
    foveal_img       = img_mosaic[:, :, h2:H, w2:W]

    # --- 2. Redimensionner à 512, 256, 128, 64 ---
    global_up       = F.interpolate(global_img, size=(512, 512), mode='bilinear', align_corners=False)
    intermediate_up = F.interpolate(intermediate_img, size=(256, 256), mode='bilinear', align_corners=False)
    perifoveal_up   = F.interpolate(perifoveal_img, size=(128, 128), mode='bilinear', align_corners=False)
    foveal_up       = F.interpolate(foveal_img, size=(64, 64), mode='bilinear', align_corners=False)

    # --- 3. Composer la superposition centrée ---
    out = global_up.clone()

    def paste_center(base, patch, attn=False):
        _, _, H, W = base.shape
        _, _, h, w = patch.shape
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        if attn:
          base[:, :, y0:y0+h, x0:x0+w] += patch
        else:
          base[:, :, y0:y0+h, x0:x0+w] = patch
        return base

    out = paste_center(out, intermediate_up, attn=attn)
    out = paste_center(out, perifoveal_up, attn=attn)
    out = paste_center(out, foveal_up, attn=attn)

    return out


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """tensor : (C, H, W) ou (B, C, H, W)"""
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (tensor.cpu().float() * s + m).clamp(0, 1)


def visualize_saccades(dataset, idx=0, n_views=5, zoom=1.5, std=0.3, seed=42):
    """
    Visualise une séquence de saccades sur une image du dataset.

    Gauche  : image originale (512×512) + trajectoire des saccades
    Droite  : n_views vues fovéales reconstruites via foveate_mosaic
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 1. Image brute (avant transform) ────────────────────────────────
    # On accède au dataset de base pour avoir la PIL originale
    img_pil, label = dataset.base[idx]
    pre    = dataset.pre                         # Resize + CenterCrop
    img_pil = pre(img_pil)                       # PIL 512×512
    img_np  = np.array(img_pil)                  # (512, 512, 3)  uint8

    # ── 2. Générer les saccades avec les mêmes paramètres ───────────────
    sz = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=n_views)
    np.random.seed(seed)
    views = sz(img_pil)   # liste de (pil_shifted, sx, sy, zoom)

    # Coordonnées des points de fixation dans l'image 512×512
    # sx, sy ∈ [-1, 1] → pixels dans l'image zoomée recadrée
    # Le centre de fixation dans l'image originale est :
    #   cx_px = 512/2 * (1 + sx/2)  (même formule que shift_zoom)
    W, H   = img_pil.size   # 512, 512
    fix_pts = []
    for _, sx, sy, _ in views:
        zoomed_ref = min(int(W * zoom), int(H * zoom))
        cx = zoomed_ref / 2 * (1 + sx / 2)
        cy = zoomed_ref / 2 * (1 + sy / 2)
        # ramener dans le repère image originale (avant zoom)
        cx_orig = cx / zoom
        cy_orig = cy / zoom
        fix_pts.append((cx_orig, cy_orig))

    # ── 3. Construire les mosaïques fovéales ─────────────────────────────
    fov_tf     = dataset.fov          # FoveatedPyramidTransform
    to_tensor  = dataset.to_tensor
    normalize  = dataset.normalize

    mosaics = []
    for pil_shifted, sx, sy, z in views:
        mosaic = fov_tf(pil_shifted)         # PIL 128×128
        mosaic = to_tensor(mosaic)           # (3,128,128)
        mosaic = normalize(mosaic)
        mosaics.append(mosaic)

    mosaics = torch.stack(mosaics).unsqueeze(0)   # (1, V, 3, 128, 128)
    # foveate_mosaic attend (B, C, H, W) — on boucle sur les vues
    fov_imgs = []
    for v in range(n_views):
        mosaic_v  = mosaics[0, v].unsqueeze(0)    # (1, 3, 128, 128)
        mosaic_d  = denormalize(mosaic_v[0]).unsqueeze(0)  # dénorm avant foveate
        fov_out   = foveate_mosaic(mosaic_d)       # (1, 3, 512, 512)
        fov_np    = fov_out[0].permute(1, 2, 0).numpy()   # (512,512,3)
        fov_np    = np.clip(fov_np, 0, 1)
        fov_imgs.append(fov_np)

    # ── 4. Figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4 + 3 * n_views, 4))
    # grille : 1 colonne pour l'original + n_views colonnes pour les vues
    gs  = fig.add_gridspec(1, 1 + n_views, wspace=0.05,
                           width_ratios=[2] + [1] * n_views)

    # ── Panneau gauche : image originale + saccades ──────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_np)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_views))

    xs_plot = [pt[0] for pt in fix_pts]
    ys_plot = [pt[1] for pt in fix_pts]

    # trajectoire
    ax0.plot(xs_plot, ys_plot,
             color='white', linewidth=1.2, alpha=0.7, zorder=2)

    # points de fixation numérotés
    for i, (px, py) in enumerate(fix_pts):
        ax0.scatter(px, py, color=colors[i], s=80, zorder=3,
                    edgecolors='white', linewidths=0.8)
        ax0.text(px + 8, py - 8, str(i + 1),
                 color=colors[i], fontsize=8, fontweight='bold',
                 zorder=4)

    # croix au centre
    ax0.axhline(H / 2, color='white', linewidth=0.5, alpha=0.3)
    ax0.axvline(W / 2, color='white', linewidth=0.5, alpha=0.3)

    ax0.set_title(f"Image originale\n(classe {label})", fontsize=9)
    ax0.axis('off')

    # légende colorbar saccades
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"saccade {i+1}  "
                       f"sx={views[i][1]:+.2f} sy={views[i][2]:+.2f}")
        for i in range(n_views)
    ]
    ax0.legend(handles=legend_handles, fontsize=6.5,
               loc='lower left', framealpha=0.6)

    # ── Panneaux droits : vues fovéales ──────────────────────────────────
    for v in range(n_views):
        ax = fig.add_subplot(gs[0, 1 + v])
        ax.imshow(fov_imgs[v])
        ax.set_title(f"Vue {v+1}\nsx={views[v][1]:+.2f}\nsy={views[v][2]:+.2f}",
                     fontsize=7.5, color=colors[v])
        # cadre coloré selon la saccade
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[v])
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Séquence de saccades fovéales", fontsize=11, y=1.02)
    plt.savefig("saccades_visualization.png", dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.show()
    return fig


@torch.no_grad()
def compute_saliency_grid(
        dino_student_y,
        dino_teacher,
        img1, img2, 
        grid_size=11,
        y_min=-5/6, y_max=5/6,
        zoom=1,
        device='cuda',
        norm=False
    ):
    device = next(dino_student_y.parameters()).device

    # --- Préparation : force batch size = 1 ---
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)
    
    img1 = img1.to(device)
    img2 = img2.to(device)

    z1_ref = dino_teacher.forward_features(img1)[:, 0, :]      # (1, 768)

    # --- Embedding teacher (vecteur de référence) ---
    z2 = dino_teacher.forward_features(img2)[:, 0, :]      # (1, 768)
    if norm:
        z2_norm = F.normalize(z2, dim=-1)


    # --- Grille 11×11 de valeurs Y ---
    xs = torch.linspace(y_min * zoom, y_max * zoom, grid_size, device=device)
    ys = torch.linspace(y_min * zoom, y_max * zoom, grid_size, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")  # X: horizontal, Y: vertical

    grid = torch.stack([X, Y], dim=-1)            # (11,11,2)  -> (x,y)
    grid_flat = grid.reshape(-1, 2)               # (121,2)

    # Répéter l’image pour 121 positions
    img1_rep = img1.expand(grid_flat.shape[0], -1, -1, -1)             # (121, 3,128,128)

    # --- Forward student sur toute la grille ---
    z1_batch, z_y_batch = dino_student_y(img1_rep, grid_flat, layernorm=False)          # z_y_batch shape (121, 768)
    if norm:
        z_y_batch_norm = F.normalize(z_y_batch, dim=-1)

    # --- Produit scalaire avec le vecteur enseignant ---
    saliency = torch.sum(z_y_batch_norm * z2_norm, dim=-1)                       # (121,)
    saliency = saliency.reshape(grid_size, grid_size)                  # (11, 11)

    return z1_ref, z_y_batch, z2, saliency #saliency, grid, z_y_batch.reshape(grid_size, grid_size, -1), z2
