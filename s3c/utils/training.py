import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import pandas as pd

from s3c.models.student_teacher import StudentWithYPredictor, update_ema_student_teacher
from s3c.data.datasets import make_pair_dataloader, make_dataloader

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from datetime import datetime


def get_parent_synset(wnid, level=2):
    """
    Remonte de `level` niveaux dans la hiérarchie WordNet.
    level=1 : parent direct
    level=2 : grand-parent (ex: 'dog' → 'canine' → 'carnivore')
    """
    synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
    for _ in range(level):
        hypernyms = synset.hypernyms()
        if not hypernyms:
            break
        synset = hypernyms[0]
    return synset



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

def _atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)

def _optim_path(ckpt_path):
    root, ext = os.path.splitext(ckpt_path)
    return root + "_optim" + ext   # X_checkpoint.pt -> X_checkpoint_optim.pt

def train_teacher_student_Xattn(
        dino_model, mlp, std_min, std_max,
        data_kwargs=None,
        start_center=True, layernorm=False,
        train_dir="", val_dir="",
        batch_size=32, device="cuda", epochs=10,
        lr=1e-4, weight_decay=1e-4, warmup_epochs=5,
        backbone_lr_scale=0.01, head_lr_scale=0.1,
        log_interval=100, val_batches=5, ema_momentum=0.999,
        save_dir="../checkpoints/checkpoints_teacher_student",
        run_name=None,            # None -> horodatage automatique
        resume_checkpoint=None,   # epoch et batch sont lus dans le checkpoint
    ):
    
    os.makedirs(save_dir, exist_ok=True)
    run_name = run_name or datetime.now().strftime("%Y%m%d")
    ckpt_path  = os.path.join(save_dir, f"{run_name}_checkpoint.pt")
    optim_path = _optim_path(ckpt_path)
    log_path   = os.path.join(save_dir, f"{run_name}_training_log.csv")

    data_kwargs = data_kwargs or {}
    val_data_kwargs = {**data_kwargs, "random_levels": False}

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dino_student_y = StudentWithYPredictor(deepcopy(dino_model)).to(device)
    dino_teacher = deepcopy(dino_model).to(device)
    if not layernorm:
        dino_teacher.norm = nn.Identity()
    for p in dino_teacher.parameters():
        p.requires_grad = False

    backbone_params, head_params = split_backbone_head_params(dino_student_y)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params,  "lr": lr * backbone_lr_scale},
            {"params": head_params,      "lr": lr * head_lr_scale},
            {"params": mlp.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    #scaler = torch.amp.GradScaler("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda")      # PyTorch >= 2.3
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler()       # PyTorch plus ancien

    # --- Chargement du checkpoint ---
    resume_epoch, resume_batch = 0, 0
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        dino_student_y.load_state_dict(ckpt["student"])
        dino_teacher.load_state_dict(ckpt["teacher"])
        mlp.load_state_dict(ckpt["mlp"])
        history = ckpt["history"]
        resume_epoch, resume_batch = ckpt["epoch"], ckpt["batch"]
        del ckpt

        optim_file = _optim_path(resume_checkpoint)
        opt = torch.load(optim_file, map_location=device) if os.path.exists(optim_file) else None
        if opt is not None and (opt["epoch"], opt["batch"]) == (resume_epoch, resume_batch):
            optimizer.load_state_dict(opt["optimizer"])
            scheduler.load_state_dict(opt["scheduler"])
            scaler.load_state_dict(opt["scaler"])
        else:
            print("État de l'optimiseur absent ou incohérent : optimiseur réinitialisé")
            for _ in range(resume_epoch):
                scheduler.step()
        del opt
        torch.cuda.empty_cache()
        print(f"Reprise à l'epoch {resume_epoch + 1}, batch {resume_batch}")
    else:
        history = {"epoch": [], "batch": [],
                   "train_loss_1": [], "val_loss_1": [],
                   "train_loss_2": [], "val_loss_2": [],
                   "train_loss_3": [], "val_loss_3": []}
    print(f"LR courante : {scheduler.get_last_lr()}")
    
    print(f"LR courante : {scheduler.get_last_lr()}")

    criterion = nn.MSELoss()
    dino_teacher.eval()

    for epoch in range(resume_epoch, epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        dino_student_y.train()
        mlp.train()
        running_loss_1 = 0.0
        running_loss_2 = 0.0
        running_loss_3 = 0.0

        mult = (1 - 0.95**epoch) / (1 - 0.95**max(epochs - 1, 1))
        std = std_min + mult * (std_max - std_min)

        train_loader = make_pair_dataloader(train_dir, std, start_center=start_center, batch_size=batch_size, **data_kwargs)
        val_loader = make_pair_dataloader(val_dir, std, start_center=start_center, batch_size=batch_size, **val_data_kwargs)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (img1, img2, y_target) in enumerate(pbar):

            # --- Skip des batches déjà traités à l'epoch de reprise ---
            if epoch == resume_epoch and batch_idx < resume_batch:
                if batch_idx % 1000 == 0:
                    print(f"  Skip batch {batch_idx}/{resume_batch}...")
                del img1, img2, y_target
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            img1, img2, y_target = img1.to(device), img2.to(device), y_target.to(device)

            

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                z1_target = dino_teacher.forward_features(img1)[:, 0, :]
                z2 = dino_teacher.forward_features(img2)[:, 0, :]
            z1_target, z2 = z1_target.float(), z2.float()   # hors inference_mode, comme avant

            #z1, z2_pred = dino_student_y(img1, y_target, layernorm=layernorm)
            with torch.autocast("cuda", dtype=torch.float16):
                z1, z2_pred = dino_student_y(img1, y_target, layernorm=layernorm)
                y_pred = mlp(z1, z2.detach().clone())

            '''loss_align = criterion(z2_pred, z2.detach().clone())
            running_loss_1 += loss_align.item()

            #y_pred = mlp(z1, z2.detach().clone())
            loss_shift = criterion(y_pred, y_target)
            running_loss_2 += loss_shift.item()

            loss_reg = criterion(z1, z1_target.detach().clone())
            running_loss_3 += loss_reg.item()'''

            loss_align = criterion(z2_pred.float(), z2.detach().clone())
            running_loss_1 += loss_align.item()

            loss_shift = criterion(y_pred.float(), y_target)
            running_loss_2 += loss_shift.item()

            loss_reg   = criterion(z1.float(), z1_target.detach().clone())
            running_loss_3 += loss_reg.item()

            loss = loss_align + loss_shift + loss_reg

            optimizer.zero_grad(set_to_none=True)
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
                    for _ in range(val_batches):
                        img1, img2, y_target = next(val_iter)
                        img1, img2, y_target = img1.to(device), img2.to(device), y_target.to(device)
                        z1, z2_pred = dino_student_y(img1, y_target, layernorm=layernorm)
                        z1_target = dino_teacher.forward_features(img1)[:, 0, :]
                        z2 = dino_teacher.forward_features(img2)[:, 0, :]
                        val_loss_1 += criterion(z2_pred, z2).item()
                        y_pred = mlp(z1, z2)
                        val_loss_2 += criterion(y_pred, y_target).item()
                        val_loss_3 += criterion(z1, z1_target).item()

                val_loss_1 /= val_batches
                val_loss_2 /= val_batches
                val_loss_3 /= val_batches
                print(f"Validation losses: {val_loss_1:.4f} / {val_loss_2:.4f} / {val_loss_3:.4f}")

                history["epoch"].append(epoch + 1)
                history["batch"].append(batch_idx + 1)
                history["train_loss_1"].append(avg_loss_1)
                history["train_loss_2"].append(avg_loss_2)
                history["train_loss_3"].append(avg_loss_3)
                history["val_loss_1"].append(val_loss_1)
                history["val_loss_2"].append(val_loss_2)
                history["val_loss_3"].append(val_loss_3)

                _atomic_save({
                    "epoch": epoch, "batch": batch_idx + 1,
                    "student": dino_student_y.state_dict(),
                    "teacher": dino_teacher.state_dict(),
                    "mlp": mlp.state_dict(),
                    "history": history,
                }, ckpt_path)

                _atomic_save({
                    "epoch": epoch, "batch": batch_idx + 1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }, optim_path)

                pd.DataFrame(history).to_csv(log_path, index=False)

                del img1, img2, y_pred, y_target
                del z1, z1_target, z2, z2_pred
                torch.cuda.empty_cache()

                dino_student_y.train()
                mlp.train()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Peak MB:", torch.cuda.max_memory_allocated() / 1024**2)
        scheduler.step()

    if os.path.exists(optim_path):
        os.remove(optim_path)
        print(f"Fichier optimiseur supprimé : {optim_path}")
    print(f"✅ Entraînement terminé.")
    return dino_student_y, dino_teacher, mlp, history





@torch.no_grad()
def _evaluate(dino_model, linear_head, loader, criterion, device, amp_dtype, use_amp, max_batches=None):
    """Loss et top-1 (%) sur max_batches batchs du loader (None = tout le loader)."""
    linear_head.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for i, (mosaics, _, _, _, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        mosaics, labels = mosaics.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            features = dino_model(mosaics)[:, 0]
        out = linear_head(features.float())
        loss_sum += criterion(out, labels).item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    linear_head.train()
    return loss_sum / total, 100.0 * correct / total
 
 
def train_linear_head(
    dino_model,
    linear_head,
    std,
    n_uplet=1,
    start_center=False,
    train_dir="",
    val_dir="",
    data_kwargs=None,
    batch_size=256,
    device="cuda",
    epochs=20,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True,
    warmup_epochs=5,
    warmup_start_factor=0.1,
    amp="float16",            # backbone gelé : "float16" | "bfloat16" | "none" (la tête reste en fp32)
    log_interval=100,
    val_batches=20,
    final_full_val=True,
    save_dir="../checkpoints/checkpoints_teacher_student",
    run_name=None,
):
    os.makedirs(save_dir, exist_ok=True)
    run_name = run_name or datetime.now().strftime("%Y%m%d")
    ckpt_path = os.path.join(save_dir, f"{run_name}_checkpoint.pt")
    log_path = os.path.join(save_dir, f"{run_name}_training_log.csv")
 
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "none": None}[amp]
    use_amp = amp_dtype is not None
 
    # backbone gelé, tête entraînable
    dino_model = dino_model.to(device).eval()
    for p in dino_model.parameters():
        p.requires_grad = False
    linear_head = linear_head.to(device).train()
 
    # SGD sur la tête seule ; warmup linéaire puis cosinus, avancé à chaque epoch
    optimizer = torch.optim.SGD([p for p in linear_head.parameters() if p.requires_grad],
                                lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=warmup_start_factor, total_iters=warmup_epochs),
                    CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()
 
    data_kwargs = {**data_kwargs, "random_levels": False}
    loader_kw = dict(std=std, n_views=n_uplet, start_center=start_center,
                     batch_size=batch_size, **(data_kwargs or {}))
    train_loader = make_dataloader(train_dir, **loader_kw)
    val_loader = make_dataloader(val_dir, **loader_kw)
 
    history = {"epoch": [], "batch": [], "train_loss": [], "val_loss": [], "val_top1": []}
 
    for epoch in range(epochs):
        running_loss = torch.zeros((), device=device)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
 
        for batch_idx, (mosaics, _, _, _, labels) in enumerate(pbar):
            mosaics, labels = mosaics.to(device, non_blocking=True), labels.to(device, non_blocking=True)
 
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                features = dino_model(mosaics)[:, 0]
            loss = criterion(linear_head(features.float()), labels)
 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()          # pas de .item() à chaque itération
 
            if (batch_idx + 1) % log_interval == 0:
                train_loss = running_loss.item() / log_interval
                running_loss.zero_()
 
                val_loss, val_top1 = _evaluate(dino_model, linear_head, val_loader, criterion,
                                               device, amp_dtype, use_amp, max_batches=val_batches)
 
                pbar.set_postfix({"loss": f"{train_loss:.4f}", "top1": f"{val_top1:.2f}%"})
                for k, v in zip(history, (epoch + 1, batch_idx + 1, train_loss, val_loss, val_top1)):
                    history[k].append(v)
                _atomic_save({"epoch": epoch, "batch": batch_idx + 1,
                             "classifier": linear_head.state_dict(), "history": history}, ckpt_path)
                pd.DataFrame(history).to_csv(log_path, index=False)
 
        scheduler.step()
 
    # validation complète finale (ajoutée à l'historique comme dernière ligne, batch = -1)
    if final_full_val:
        val_loss, val_top1 = _evaluate(dino_model, linear_head, val_loader, criterion,
                                       device, amp_dtype, use_amp)
        print(f"Validation complète : loss={val_loss:.4f} | top-1={val_top1:.2f}%")
        for k, v in zip(history, (epochs, -1, float("nan"), val_loss, val_top1)):
            history[k].append(v)
 
    _atomic_save({"epoch": epochs - 1, "batch": len(train_loader),
                 "classifier": linear_head.state_dict(), "history": history}, ckpt_path)
    pd.DataFrame(history).to_csv(log_path, index=False)
    return linear_head, history