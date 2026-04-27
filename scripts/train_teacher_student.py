"""
Script d'entraînement teacher-student SMILES.

Usage :
    python scripts/train_teacher_student.py --config configs/teacher_student.yaml
    python scripts/train_teacher_student.py --config configs/teacher_student.yaml \
        training.epochs=50 training.lr=5e-5
"""
import argparse
import yaml
import torch
from pathlib import Path

# permettre l'override CLI de paramètres YAML
def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        keys, val = override.split("=")
        keys = keys.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # cast automatique
        try:    val = int(val)
        except: 
            try: val = float(val)
            except: pass
        d[keys[-1]] = val
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")  # training.epochs=50
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # imports locaux après config
    import timm
    from s3c.models.foveated_vit import FoveatedMultiViT, build_foveated_pos_embed
    from s3c.models.student_teacher import StudentWithYPredictor, update_ema_student_teacher
    from s3c.models.heads import ShiftPredictor
    from s3c.data.datasets import FoveatedPairDataset
    from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform
    from s3c.utils.training import train_teacher_student_Xattn

    device = cfg["training"]["device"]

    # ── modèle ───────────────────────────────────────────────────────────
    base_model = timm.create_model(
        cfg["model"]["name"], pretrained=cfg["model"]["pretrained"]
    )
    base_model.patch_embed.strict_img_size = False
    build_foveated_pos_embed(
        base_model,
        base_img_size         = cfg["model"]["img_size"],
        patch_size            = cfg["model"]["patch_size"],
        mosaic_sub_patch_grid = cfg["model"]["mosaic_sub_patch_grid"],
        scales                = cfg["model"]["scales"],
    )

    fovea_transform = FoveatedPyramidTransform(
        output_size=cfg["model"]["mosaic_size"]
    )
    mlp = ShiftPredictor(emb_dim=768, hidden_dim=512).to(device)

    # ── entraînement ─────────────────────────────────────────────────────
    trained = train_teacher_student_Xattn(
        dino_model   = base_model,
        mlp          = mlp,
        zoom         = cfg["data"]["zoom"],
        std_min      = cfg["training"]["std_min"],
        std_max      = cfg["training"]["std_max"],
        resolution   = cfg["model"]["mosaic_size"],
        train_dir    = cfg["data"]["train_dir"],
        val_dir      = cfg["data"]["val_dir"],
        batch_size   = cfg["training"]["batch_size"],
        device       = device,
        epochs       = cfg["training"]["epochs"],
        lr           = cfg["training"]["lr"],
        log_interval = cfg["training"]["log_interval"],
        ema_momentum = cfg["training"]["ema_momentum"],
        save_dir     = cfg["training"]["save_dir"],
    )


if __name__ == "__main__":
    main()