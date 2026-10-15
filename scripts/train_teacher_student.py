import argparse
from pathlib import Path
from datetime import datetime


import yaml

from s3c.utils.config import load_config  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="ex: training.epochs=50")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    print(yaml.dump(cfg, sort_keys=False))

    # imports locaux après lecture de la config
    import timm
    from s3c.models.foveated_vit import build_foveated_pos_embed, build_flat_pos_embed  
    from s3c.models.heads import ShiftPredictor
    from s3c.utils.training import train_teacher_student_Xattn

    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    transform = m["transform"]
    if transform not in ("fovea", "flat"):
        raise ValueError(f"model.transform doit valoir 'fovea' ou 'flat', reçu : {transform!r}")
    run_name = str(t["run_name"]) if t["run_name"] else datetime.now().strftime("%H-%M-%S")
    device = t["device"]

    # ── sauvegarde de la config effectivement utilisée ───────────────────
    save_dir = Path(t["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{run_name}_config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # ── modèle ───────────────────────────────────────────────────────────
    base_model = timm.create_model(m["name"], pretrained=m["pretrained"])
    base_model.patch_embed.strict_img_size = False
    if transform == "fovea":
        build_foveated_pos_embed(
            base_model,
            base_img_size         = m["img_size"],
            patch_size            = m["patch_size"],
            mosaic_sub_patch_grid = m["mosaic_sub_patch_grid"],
            scales                = m["scales"],
        )
    else:  # "flat"
        build_flat_pos_embed(
            base_model,
            base_img_size=224,
            patch_size=8,
            dest_image_size = m["mosaic_size"],
        )

    mlp = ShiftPredictor(
        emb_dim    = m["emb_dim"],
        hidden_dim = m["head_hidden_dim"],
    ).to(device)

    data_kwargs = dict(
        transform     = m["transform"],
        resolution    = m["mosaic_size"],
        num_workers   = d["num_workers"],
        limit         = d["limit"],
        n_levels      = len(m["scales"]),
        random_levels = t["random_levels"],
        fixed_level   = t["fixed_level"],
    )

    # ── entraînement ─────────────────────────────────────────────────────
    trained = train_teacher_student_Xattn(
        dino_model        = base_model,
        mlp               = mlp,        
        zoom              = t["zoom"],
        std_min           = float(t["std_min"]) / t["zoom"],
        std_max           = float(t["std_max"]) / t["zoom"],
        data_kwargs       = data_kwargs,
        start_center      = t["start_center"],
        layernorm         = m["layernorm"],
        train_dir         = d["train_dir"],
        val_dir           = d["val_dir"],
        batch_size        = t["batch_size"],
        device            = device,
        epochs            = t["epochs"],
        lr                = float(t["lr"]),
        weight_decay      = float(t["weight_decay"]),
        warmup_epochs     = t["warmup_epochs"],
        backbone_lr_scale = float(t["backbone_lr_scale"]),
        head_lr_scale     = float(t["head_lr_scale"]),
        log_interval      = t["log_interval"],
        val_batches       = t["val_batches"],
        ema_momentum      = float(t["ema_momentum"]),
        save_dir          = str(save_dir),
        run_name          = run_name,
        resume_checkpoint = t["resume_checkpoint"],
    )


if __name__ == "__main__":
    main()