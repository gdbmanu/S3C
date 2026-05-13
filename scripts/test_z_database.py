# %%

import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


from s3c.data.datasets import ImageNetZDataset

# --- Configuration générale ---
# data_dir = val_dir = "/home/INT/dauce.e/data/Imagenet_full/val"   # Imagenet Validation set
batch_size = 56
num_workers = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resolution = 128
embed_dim = 768

n_uplet = 30

save_dir = "data/Imagenet_Z"

zoom = 1.5

std = 0.5 / zoom

img_train_dir = "~/data/Imagenet_full/train"   # Imagenet Validation set
img_val_dir = "~/data/Imagenet_full/val"   # Imagenet Validation set

train_dataset_raw   = datasets.ImageFolder(img_train_dir, transform=None)
val_dataset_raw   = datasets.ImageFolder(img_val_dir, transform=None)


 # ── lancement ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # test du dataset de chargement
    ds_train = ImageNetZDataset(f"{save_dir}/train")
    ds_val = ImageNetZDataset(f"{save_dir}/val")
    print(f"\nTrain : {len(ds_train)} embeddings")
    print(f"Imagenet train : {len(train_dataset_raw)} images")
    print(f"Val   : {len(ds_val)} embeddings")
    print(f"Imagenet val : {len(val_dataset_raw)} images")

    num_unique_labels = len(ds_train.classes)
    print(f"Nombre de classes (labels) : {num_unique_labels}")

    zs, xs, ys, label = ds_train[0]
    print(f"Shape embedding : {zs.shape}")   # (n_uplet, 768)
    print(f"xs           : {xs}")      
    print(f"ys           : {ys}")      
    print(f"Label           : {label}")      



