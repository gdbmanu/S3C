"""
Test "une saccade" : comparaison centre vs centre+saccade prédite par ist_transformer.

Pipeline par batch :
  1. vue centrale (0,0)            -> teacher_model -> z_center
  2. set de features pré-enregistré (même image, base features_path)
       -> ist_transformer          -> z_star
  3. shift_predictor(z_center, z_star) -> (x_star, y_star)
  4. vue saccadée en (x_star, y_star) -> teacher_model -> z_sacc
  5. classification (norm_ref + z_linear_head) : centre / saccade / somme des logits
"""
import argparse
import copy
import os
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as FF
from tqdm import tqdm

from s3c.models.foveated_vit import build_foveated_pos_embed
from s3c.models.heads import ShiftPredictor, ShiftPredictor2,  WhatTransformer # TODO: classe de ist_transformer
from s3c.data.transforms import FoveatedPyramidTransform

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# Données : images ImageNet + sets de features, appariés par nom de fichier
# ─────────────────────────────────────────────────────────────────────────────
class PairedImageFeatureDataset(Dataset):
    """img_root/<classe>/xxx.JPEG  <->  feat_root/<classe>/xxx.pt
    Renvoie (PIL image, features [N,768], label, chemin relatif)."""

    def __init__(self, img_root, feat_root, strict=True):
        self.img_root, self.feat_root = Path(img_root), Path(feat_root)
        classes = sorted(d.name for d in self.img_root.iterdir() if d.is_dir())
        class_to_idx = {c: i for i, c in enumerate(classes)}  # = ImageFolder

        self.samples, missing = [], []
        for p in sorted(self.img_root.rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            rel = p.relative_to(self.img_root)
            fp = (self.feat_root / rel).with_suffix(".pt")
            if not fp.exists():
                missing.append(str(rel))
                continue
            self.samples.append((p, fp, class_to_idx[rel.parts[0]], str(rel)))

        if missing:
            msg = f"{len(missing)} images sans .pt (ex: {missing[:3]})"
            if strict:
                raise FileNotFoundError(msg)
            print("WARNING:", msg)
        print(f"{len(self.samples)} paires image/features")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, feat_p, label, rel = self.samples[i]
        img = Image.open(img_p).convert("RGB")
        # le .pt est un dict {"zs": (n_sac, D), "xs": ..., "ys": ...} : seul zs est utilisé ici
        feats = torch.load(feat_p, map_location="cpu")["zs"].float()
        return img, feats, label, rel


def collate_paired(batch):
    imgs, feats, labels, names = zip(*batch)
    return list(imgs), torch.stack(feats), torch.tensor(labels), list(names)


# ─────────────────────────────────────────────────────────────────────────────
# Outils vision
# ─────────────────────────────────────────────────────────────────────────────
def shift_zoom(img, x, y, zoom):
    """Zoom puis crop (taille = petit côté original) centré sur (x,y) ∈ [-1,1]."""
    w, h = img.size
    size_ref = min(w, h)
    w_z, h_z = int(w * zoom), int(h * zoom)
    zoomed_ref = min(w_z, h_z)

    img_z = FF.resize(img, (h_z, w_z), interpolation=Image.BICUBIC)
    cx = w_z / 2 + zoomed_ref * x / 2
    cy = h_z / 2 + zoomed_ref * y / 2

    left = int(cx - size_ref / 2)
    top = int(cy - size_ref / 2)
    right, bottom = left + size_ref, top + size_ref

    pl, pt = max(0, -left), max(0, -top)
    pr, pb = max(0, right - w_z), max(0, bottom - h_z)
    if any([pl, pt, pr, pb]):
        img_z = FF.pad(img_z, (pl, pt, pr, pb), padding_mode="reflect")
    left += pl
    top += pt
    return img_z.crop((left, top, left + size_ref, top + size_ref))


def make_mosaics(imgs, xs, ys, zoom, fovea_transform):
    return torch.stack([
        fovea_transform(shift_zoom(img, float(x), float(y), zoom))
        for img, x, y in zip(imgs, xs, ys)
    ])


def select_context(feats, k):
    """Tire k vecteurs au hasard (sans remise) parmi les N du set. feats: [B,N,D]."""
    B, N, D = feats.shape
    idx = torch.rand(B, N, device=feats.device).argsort(dim=1)[:, :k]
    return torch.gather(feats, 1, idx.unsqueeze(-1).expand(-1, -1, D))


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ─────────────────────────────────────────────────────────────────────────────
def load_state(module, state_dict, name, strict=False):
    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    print(f"[{name}] manquants: {missing} | inattendus: {unexpected}")
    return module


def build_models(args, device, n_classes=1000):
    foveated_kw = dict(base_img_size=224, patch_size=8, mosaic_sub_patch_grid=8,
                       scales=(1.0, 0.5, 0.25, 0.125))

    # LayerNorm de référence, appliqué avant la tête de classification
    ref = timm.create_model(args.model_name, pretrained=True)
    ref.patch_embed.strict_img_size = False
    build_foveated_pos_embed(ref, **foveated_kw)
    norm_ref = copy.deepcopy(ref.norm).to(device).eval()
    embed_dim = ref.embed_dim
    del ref

    # teacher (sans LayerNorm finale)
    ckpt = torch.load(os.path.join(args.model_dir, f"checkpoint_epoch{args.epoch_teacher}.pt"),
                      map_location="cpu")
    teacher = timm.create_model(args.model_name, pretrained=True)
    teacher.patch_embed.strict_img_size = False
    teacher.norm = nn.Identity()
    build_foveated_pos_embed(teacher, **foveated_kw)
    load_state(teacher, ckpt["teacher"], "teacher")
    teacher.to(device).eval()

    # shift_predictor (clé "mlp" du checkpoint) 260924_SHIFT__a3e-06
    shift_ckpt = torch.load(os.path.join(args.shift_dir, f"checkpoint_epoch100.pt"),
                      map_location="cpu")
    shift_predictor = ShiftPredictor2(emb_dim=embed_dim, hidden_dim=512)
    load_state(shift_predictor, shift_ckpt["shift_predictor"], "shift_predictor")
    #shift_predictor = ShiftPredictor(emb_dim=embed_dim, hidden_dim=512)
    #load_state(shift_predictor, ckpt["mlp"], "shift_predictor")
    shift_predictor.to(device).eval()

    # ist_transformer
    # TODO : adapter le constructeur et la clé du checkpoint à ton modèle
    ist_transformer = WhatTransformer(emb_dim=embed_dim, n_classes=n_classes,residual = True, skip_ff_l=True)
    ist_ckpt = torch.load(os.path.join(args.ist_ckpt, "checkpoint_epoch100.pt"), 
                          map_location="cpu")
    load_state(ist_transformer, ist_ckpt.get("ist_transformer", ist_ckpt), "ist_transformer", strict=True)
    ist_transformer.to(device).eval()

    # tête de classification
    head = nn.Linear(embed_dim, 1000)
    head_ckpt = torch.load(os.path.join(args.head_dir, f"checkpoint_epoch{args.epoch_head}.pt"),
                           map_location="cpu")
    load_state(head, head_ckpt["classifier"], "z_linear_head")
    head.to(device).eval()

    return teacher, norm_ref, shift_predictor, ist_transformer, head


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_saccade(teacher, norm_ref, shift_predictor, ist_transformer, z_linear_head,
                     loader, fovea_transform, zoom, k_ctx, device, out_path):
    correct = {"center": 0, "saccade": 0, "total": 0}
    n = 0
    
    cos_sum = {"z_vs_feats": 0.0, "z_vs_feats_shuf": 0.0, "z_vs_zstar": 0.0}
 
    def summary():
        acc = {k: 100 * v / max(n, 1) for k, v in correct.items()}
        return {"acc_center": acc["center"], "acc_saccade": acc["saccade"],
                "acc_total": acc["total"], "delta": acc["saccade"] - acc["center"],
                "delta_total": acc["total"] - acc["center"], "n_images": n,
                **{f"cos_{k}": v / max(n, 1) for k, v in cos_sum.items()}}

    pbar = tqdm(loader, desc="Évaluation saccade")
    for i, (imgs, feats, labels, _names) in enumerate(pbar):
        labels, feats = labels.to(device), feats.to(device)
        B = labels.shape[0]

        # 1. vue centrale -> z_center
        mos_c = make_mosaics(imgs, [0.0] * B, [0.0] * B, zoom, fovea_transform).to(device)
        z_center = teacher(mos_c)                       # (B, D)  (avant LayerNorm)
        logits_center = z_linear_head(norm_ref(z_center))
        #logits_center = z_linear_head(z_center)

        # 2. set de features pré-enregistré -> z_star
        z_star = ist_transformer(select_context(feats, k_ctx), None).squeeze(1)   # (B, D)

        if i == 0:  # debug : formes et plages de valeurs (à retirer ensuite)
            print(f"\n[debug] z_center {tuple(z_center.shape)}  z_star {tuple(z_star.shape)}  feats {tuple(feats.shape)}")
            cs = nn.functional.cosine_similarity
            print(f"[debug] cos(z_star, feats)={cs(z_star[:, None], feats, dim=-1).mean():.3f}  "
                  f"|z_star|={z_star.norm(dim=-1).mean():.2f} |z_center|={z_center.norm(dim=-1).mean():.2f} "
                  f"|feats|={feats.norm(dim=-1).mean():.2f}")
        assert z_star.shape == z_center.shape, \
            f"z_star {tuple(z_star.shape)} != z_center {tuple(z_center.shape)} : cosine/shift faussés par broadcasting"

        # --- contrôle de l'appariement image <-> features ---
        # cos(z_center, chaque vecteur du set) : moyenne sur les N vecteurs
        cos = nn.functional.cosine_similarity
        cos_sum["z_vs_feats"] += cos(z_center[:, None], feats, dim=-1).mean(1).sum().item()
        # baseline : mêmes z_center face aux features d'une AUTRE image du batch
        cos_sum["z_vs_feats_shuf"] += cos(z_center[:, None], feats.roll(1, dims=0), dim=-1).mean(1).sum().item()
        cos_sum["z_vs_zstar"] += cos(z_center, z_star, dim=-1).sum().item()

        # 3. saccade prédite (x_star, y_star) à partir de (z_center, z_star)
        shift = shift_predictor(z_center, z_star)       # (B, 2)
        x_star = shift[:, 0] #/ zoom
        y_star = shift[:, 1] #/ zoom

        # 4. vue saccadée -> teacher
        mos_s = make_mosaics(imgs, x_star.tolist(), y_star.tolist(), zoom, fovea_transform).to(device)
        z_sacc = teacher(mos_s)
        logits_sacc = z_linear_head(norm_ref(z_sacc))
        logits_star = z_linear_head(norm_ref(z_star))
        #logits_sacc = z_linear_head(z_sacc)

        # 5. précisions
        correct["center"] += (logits_center.argmax(1) == labels).sum().item()
        correct["saccade"] += (logits_sacc.argmax(1) == labels).sum().item()
        correct["total"] += ((logits_center + logits_sacc + logits_star).argmax(1) == labels).sum().item()
        n += B

        s = summary()
        pbar.set_postfix({"center": f"{s['acc_center']:.2f}%",
                          "saccade": f"{s['acc_saccade']:.2f}%",
                          "total": f"{s['acc_total']:.2f}%",
                          # check temporaire (non sauvegardé)
                          "cos_feats": f"{cos_sum['z_vs_feats'] / n:.3f}",
                          "cos_shuf": f"{cos_sum['z_vs_feats_shuf'] / n:.3f}",
                          "cos_zstar": f"{cos_sum['z_vs_zstar'] / n:.3f}"})
        if i % 10 == 9:
            torch.save(s, out_path)

    s = summary()
    torch.save(s, out_path)
    print("\n── Résultats ──────────────────────────────")
    for k, v in s.items():
        print(f"  {k:12s}: {v:.2f}" if k != "n_images" else f"  {k:12s}: {v}")
    return s


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_path", default="~/data/Imagenet_full")
    p.add_argument("--features_path", default="data/Imagenet_Z")
    p.add_argument("--split", default="val")
    p.add_argument("--model_name", default="vit_base_patch8_224.dino")
    p.add_argument("--model_dir", default="../checkpoints/checkpoints_EMA_Xattn_260416")
    p.add_argument("--epoch_teacher", type=int, default=20)
    p.add_argument("--head_dir", default="../checkpoints/checkpoints_260414_EMA_Xattn_1_view")
    p.add_argument("--epoch_head", type=int, default=20)
    p.add_argument("--ist_ckpt", default="../checkpoints/260925_W_sab2__a1e-05_d3e-06_mask0.2_POS_SUP_SMOOTH0.8_INDEX_RESID_SKIP_t15_space")
    #p.add_argument("--ist_ckpt", default="../checkpoints/260925_W_sab2__a1e-05_d3e-06_mask0.2_POS_SUP_SMOOTH0.5_INDEX_SYNSET4_RESID_SKIP_t15_space")
    p.add_argument("--shift_dir", default="../checkpoints/260924_SHIFT__a3e-06")
    p.add_argument("--k_ctx", type=int, default=15)
    p.add_argument("--zoom", type=float, default=1.5)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="teacher_one_saccade_forward.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    fovea_transform = transforms.Compose([
        FoveatedPyramidTransform(output_size=args.resolution, to_torch=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PairedImageFeatureDataset(Path(args.img_path).expanduser() / args.split,
                                        Path(args.features_path) / args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_paired)

    models = build_models(args, device) #, n_classes=109)
    evaluate_saccade(*models, loader, fovea_transform, args.zoom, args.k_ctx, device, args.out)


if __name__ == "__main__":
    main()
