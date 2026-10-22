import argparse
from datetime import datetime
from pathlib import Path

import yaml

from s3c.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="ex: training.epochs=50")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    print(yaml.dump(cfg, sort_keys=False))

    ckpt_path = Path(cfg["teacher"]["checkpoint"] or "").expanduser()
    #if not ckpt_path.is_file():
    #    raise FileNotFoundError(f"teacher.checkpoint introuvable : {ckpt_path}")

    # imports locaux après lecture de la config
    import timm
    import torch
    import torch.nn as nn
    from s3c.models.foveated_vit import FoveatedMultiViT, build_foveated_pos_embed, build_flat_pos_embed
    from s3c.utils.training import train_linear_head

    m, d, t = cfg["model"], cfg["data"], cfg["training"]
    transform = m["transform"]
    if transform not in ("fovea", "flat"):
        raise ValueError(f"model.transform doit valoir 'fovea' ou 'flat', reçu : {transform!r}")
    if t["fixed_level"] is not None and not 0 <= t["fixed_level"] < len(m["scales"]):
        raise ValueError(f"training.fixed_level doit être dans [0, {len(m['scales']) - 1}] ou null")
    device = t["device"]

    # ── nom du run et dossier de sortie : le dossier du teacher ──────────
    # fichiers : <run du teacher>_linear_head_<date>_{checkpoint.pt, training_log.csv, config.yaml}
    suffix = "_checkpoint.pt"
    teacher_run = ckpt_path.name[: -len(suffix)] if ckpt_path.name.endswith(suffix) else ckpt_path.stem
    head_date = str(t["run_name"]) if t["run_name"] else datetime.now().strftime("%Y%m%d")

    run_name = f"{teacher_run}_linear_head_{head_date}" if not t["orig"] else f"orig_linear_head_{head_date}"

    save_dir = Path(t["save_dir"]).expanduser() if t["save_dir"] else ckpt_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / f"{run_name}_checkpoint.pt").exists():
        raise FileExistsError(f"{run_name} existe déjà : changez training.run_name pour ne pas l'écraser.")

    # ── sauvegarde de la config effectivement utilisée ───────────────────
    with open(save_dir / f"{run_name}_config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # ── backbone : identique à train_teacher_student.py ──────────────────
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
            base_img_size   = m["img_size"],
            patch_size      = m["patch_size"],
            dest_image_size = m["mosaic_size"],
        )

    # ── teacher : poids du réseau entraîné ───────────────────────────────
    teacher = FoveatedMultiViT(base_model, norm=False)   # sans norme finale : elle est placée dans la tête
    if not t["orig"]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = teacher.model.load_state_dict(ckpt["teacher"], strict=False)
        if unexpected or any(not k.startswith("norm.") for k in missing):
            raise RuntimeError(f"Poids du teacher incompatibles : manquants={missing}, inattendus={unexpected}")
        print(f"Teacher chargé : {ckpt_path.name} (epoch {ckpt['epoch'] + 1})")

    # ── tête linéaire = LayerNorm + projection ───────────────────────────
    # La LayerNorm finale du backbone passe dans la tête (même résultat sur le token CLS) :
    # c'est celle du teacher si layernorm=True à l'entraînement, sinon celle de DINO pré-entraîné.
    norm = base_model.norm
    for p in norm.parameters():
        p.requires_grad = False
    linear_head = nn.Sequential(norm, nn.Linear(base_model.embed_dim, t["num_classes"]))

    data_kwargs = dict(
        transform   = transform,
        resolution  = m["mosaic_size"],
        num_workers = d["num_workers"],
        limit       = d["limit"],
        fixed_level = t["fixed_level"],
        zoom        = t["zoom"],
    )

    # ── entraînement ─────────────────────────────────────────────────────
    trained = train_linear_head(
        dino_model          = teacher,
        linear_head         = linear_head,
        std                 = float(d["std"]),
        n_uplet             = d["n_uplet"],
        start_center        = t["start_center"],
        train_dir           = str(Path(d["train_dir"]).expanduser()),
        val_dir             = str(Path(d["val_dir"]).expanduser()),
        data_kwargs         = data_kwargs,
        batch_size          = t["batch_size"],
        device              = device,
        epochs              = t["epochs"],
        lr                  = float(t["lr"]),
        momentum            = float(t["momentum"]),
        weight_decay        = float(t["weight_decay"]),
        nesterov            = t["nesterov"],
        warmup_epochs       = t["warmup_epochs"],
        warmup_start_factor = float(t["warmup_start_factor"]),
        amp                 = t["amp"],
        log_interval        = t["log_interval"],
        val_batches         = t["val_batches"],
        final_full_val      = t["final_full_val"],
        save_dir            = str(save_dir),
        run_name            = run_name,
    )


if __name__ == "__main__":
    main()