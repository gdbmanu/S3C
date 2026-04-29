import os
from copy import deepcopy

import torch
import torch.nn as nn

from tqdm import tqdm

import pandas as pd

from s3c.models.student_teacher import StudentWithYPredictor, update_ema_student_teacher
from s3c.data.transforms import FoveatedPyramidTransform
from s3c.data.datasets import make_dataloader

from torch.optim.lr_scheduler import LinearLR


def split_backbone_head_params(model):
    backbone = []
    head = []

    # ---- backbone global
    backbone += list(model.patch_embed.parameters())
    backbone.append(model.cls_token)
    backbone.append(model.pos_embed)
    backbone += list(model.norm.parameters())

    # ---- blocs
    for blk in model.blocks:
        backbone += list(blk.attn_x.parameters())
        backbone += list(blk.mlp_x.parameters())
        backbone += list(blk.norm1_x.parameters())
        backbone += list(blk.norm2_x.parameters())

        # head / Y branch
        head += list(blk.norm1_y.parameters())
        head += list(blk.norm2_y.parameters())
        head += list(blk.mlp_y.parameters())
        head += list(blk.cross_attn.parameters())

    # ---- têtes explicites
    head += list(model.y_proj.parameters())
    head += list(model.norm_y.parameters())

    return backbone, head


def freeze_backbone(student):
    # patch embedding
    for p in student.patch_embed.parameters():
        p.requires_grad = False

    # cls token & pos embed
    student.cls_token.requires_grad = False
    student.pos_embed.requires_grad = False

    # transformer blocks (branche image uniquement)
    for blk in student.blocks:
        for p in blk.attn_x.parameters():
            p.requires_grad = False
        for p in blk.norm1_x.parameters():
            p.requires_grad = False
        for p in blk.norm2_x.parameters():
            p.requires_grad = False
        for p in blk.mlp_x.parameters():
            p.requires_grad = False

    # norm final
    for p in student.norm.parameters():
        p.requires_grad = False

def unfreeze_backbone(student):
    for p in student.parameters():
        p.requires_grad = True

def train_teacher_student_Xattn(
    dino_model,
    mlp,
    zoom,
    std_min,
    std_max,
    resolution=128,
    start_center = True,
    layernorm    = False,
    train_dir="",
    val_dir="",
    batch_size=32,
    device='cuda',
    epochs=10,
    lr=1e-4,
    log_interval=100,
    ema_momentum=0.999,
    alpha=0.3,
    lambda_sparse=1e-2,
    save_dir="../checkpoints/checkpoints_teacher_student",
    resume_epoch=0,       # resume from crash
    resume_batch=0,       # 
    resume_checkpoint=None  # 
):
    os.makedirs(save_dir, exist_ok=True)

    dino_student_y = StudentWithYPredictor(deepcopy(dino_model)).to(device)
    dino_teacher = deepcopy(dino_model).to(device)
    if not layernorm:
        dino_teacher.norm = nn.Identity()
    for p in dino_teacher.parameters():
        p.requires_grad = False

    backbone_params, head_params = split_backbone_head_params(dino_student_y)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.01},
            {"params": head_params, "lr": lr * 0.1},
            {"params": mlp.parameters(), "lr": lr},
        ],
        weight_decay=1e-4
    )

    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)

    # --- Chargement du checkpoint ---
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        dino_student_y.load_state_dict(checkpoint["student"])
        dino_teacher.load_state_dict(checkpoint["teacher"])
        mlp.load_state_dict(checkpoint["mlp"])
        history = checkpoint["history"]
        del checkpoint
        torch.cuda.empty_cache()
        print(f"✅ Checkpoint chargé : {resume_checkpoint}")
        print(f"   Reprise à epoch {resume_epoch + 1}, batch {resume_batch}")
    else:
        history = {"epoch": [], "batch": [],
                   "train_loss_1": [], "val_loss_1": [],
                   "train_loss_2": [], "val_loss_2": [],
                   "train_loss_3": [], "val_loss_3": []}

    # Remettre le scheduler dans le bon état
    for _ in range(resume_epoch):
        scheduler.step()
    print(f"LR courante : {scheduler.get_last_lr()}")

    criterion = nn.MSELoss()
    dino_teacher.eval()

    for epoch in range(resume_epoch, epochs):
        torch.cuda.reset_peak_memory_stats()
        dino_student_y.train()
        mlp.train()
        running_loss_1 = 0.0
        running_loss_2 = 0.0
        running_loss_3 = 0.0

        mult = (1 - 0.95 ** epoch) / (1 - 0.95 ** (epochs - 1))
        std = std_min + mult * (std_max - std_min)

        train_loader = make_dataloader(train_dir, zoom, std, start_center=start_center, batch_size=batch_size)
        val_loader = make_dataloader(val_dir, zoom, std, start_center=start_center, batch_size=batch_size)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (img1, img2, y_target) in enumerate(pbar):

            # --- Skip des batches déjà traités à l'epoch de reprise ---
            if epoch == resume_epoch and batch_idx < resume_batch:
                if batch_idx % 1000 == 0:
                    print(f"  Skip batch {batch_idx}/{resume_batch}...")
                del img1, img2, y_target
                torch.cuda.empty_cache()
                continue

            img1, img2, y_target = img1.to(device), img2.to(device), y_target.to(device)

            with torch.inference_mode():
                z1_target = dino_teacher.forward_features(img1)[:, 0, :]
                z2 = dino_teacher.forward_features(img2)[:, 0, :]

            z1, z2_pred = dino_student_y(img1, y_target, layernorm=layernorm)

            loss_align = criterion(z2_pred, z2.detach().clone())
            running_loss_1 += loss_align.item()

            y_pred = mlp(z1, z2.detach().clone())
            loss_shift = criterion(y_pred, y_target)
            running_loss_2 += loss_shift.item()

            loss_reg = criterion(z1, z1_target.detach().clone())
            running_loss_3 += loss_reg.item()

            loss = loss_align + loss_shift + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            update_ema_student_teacher(dino_student_y, dino_teacher, momentum=ema_momentum)

            del img1, img2, y_pred, y_target
            del z1, z1_target, z2, z2_pred
            del loss_align, loss_shift, loss_reg

            if ((batch_idx + 1) % log_interval == 0):
                avg_loss_1 = running_loss_1 / log_interval
                avg_loss_2 = running_loss_2 / log_interval
                avg_loss_3 = running_loss_3 / log_interval
                running_loss_1 = 0.0
                running_loss_2 = 0.0
                running_loss_3 = 0.0
                pbar.set_postfix({"loss 1/2": f"{avg_loss_1:.4f} / {avg_loss_2:.4f} / {avg_loss_3:.4f}"})

                dino_student_y.eval()
                mlp.eval()
                val_loss_1 = val_loss_2 = val_loss_3 = 0.0
                val_iter = iter(val_loader)
                with torch.inference_mode():
                    for _ in range(5):
                        img1, img2, y_target = next(val_iter)
                        img1, img2, y_target = img1.to(device), img2.to(device), y_target.to(device)
                        z1, z2_pred = dino_student_y(img1, y_target, layernorm=layernorm)
                        z1_target = dino_teacher.forward_features(img1)[:, 0, :]
                        z2 = dino_teacher.forward_features(img2)[:, 0, :]
                        val_loss_1 += criterion(z2_pred, z2).item()
                        y_pred = mlp(z1, z2)
                        val_loss_2 += criterion(y_pred, y_target).item()
                        val_loss_3 += criterion(z1, z1_target).item()

                val_loss_1 /= 5
                val_loss_2 /= 5
                val_loss_3 /= 5
                print(f"Validation losses: {val_loss_1:.4f} / {val_loss_2:.4f} / {val_loss_3:.4f}")

                history["epoch"].append(epoch + 1)
                history["batch"].append(batch_idx + 1)
                history["train_loss_1"].append(avg_loss_1)
                history["train_loss_2"].append(avg_loss_2)
                history["train_loss_3"].append(avg_loss_3)
                history["val_loss_1"].append(val_loss_1)
                history["val_loss_2"].append(val_loss_2)
                history["val_loss_3"].append(val_loss_3)

                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx + 1,   # <-- sauvegardé aussi
                    "student": dino_student_y.state_dict(),
                    "teacher": dino_teacher.state_dict(),
                    "mlp": mlp.state_dict(),
                    "history": history,
                }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))

                df = pd.DataFrame(history)
                df.to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

                del img1, img2, y_pred, y_target
                del z1, z1_target, z2, z2_pred
                torch.cuda.empty_cache()

                dino_student_y.train()
                mlp.train()

        torch.cuda.empty_cache()
        print("Peak MB:", torch.cuda.max_memory_allocated() / 1024**2)
        scheduler.step()

    print(f"✅ Entraînement terminé.")
    return dino_student_y, dino_teacher, mlp, history