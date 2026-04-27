import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform

import numpy as np

import random

class FoveatedPairDataset(Dataset):
    def __init__(self, 
                 root, 
                 zoom, 
                 std, 
                 fovea_transform, 
                 resize=512,
                 crop=512,
                 start_center=True, 
                 preprocess=None, 
                 limit=None):
        self.pre_process = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
        ])
        self.base_dataset = datasets.ImageFolder(root, transform=None)
        self.shiftzoom_transform = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=2, start_center=start_center)
        self.fovea_transform = fovea_transform
        self.post_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        if limit:
            self.base_dataset.samples = self.base_dataset.samples[:limit]

    @torch.no_grad()
    def __getitem__(self, idx):          
        max_retries = 10
        for attempt in range(max_retries):
            try:
                img, _ = self.base_dataset[idx]

                # pre-processing
                img = self.pre_process(img)                  # PIL 512×512

                # 2 views only
                views = self.shiftzoom_transform(img)
                # views : (img_shifted, sx, sy, zoom)) 

                img1, x1, y1, _ = views[0]
                img2, x2, y2, _ = views[1]
                shift = torch.tensor([x2 - x1, y2 - y1], dtype=torch.float32)
                
                img1 = transforms.Resize(512)(img1)
                img1 = transforms.CenterCrop(512)(img1)
                img2 = transforms.Resize(512)(img2)
                img2 = transforms.CenterCrop(512)(img2)

                # appliquer fovéation + post_process
                level = np.random.randint(4)
                img1 = self.post_process(self.fovea_transform(img1, level=level))
                img2 = self.post_process(self.fovea_transform(img2))

                return img1, img2, shift

            except (OSError, IOError) as e:
                print(f"[Dataset] Erreur accès idx={idx}, tentative {attempt+1}/{max_retries} : {e}")
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Impossible de charger un exemple après {max_retries} tentatives.")
    

"""class FoveatedUpletDataset(Dataset):
    def __init__(self, root, shiftzoom_transform, fovea_transform,  preprocess=None, limit=None):
        self.base_dataset = datasets.ImageFolder(root, transform=None)
        self.shiftzoom_transform = shiftzoom_transform
        self.n_uplet = self.shiftzoom_transform.n_uplet
        self.fovea_transform = fovea_transform
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        if limit:
            self.base_dataset.samples = self.base_dataset.samples[:limit]

    @torch.no_grad()
    def __getitem__(self, idx):
        img, y = self.base_dataset[idx]
        imgs = self.shiftzoom_transform(img)

        for i in range(self.n_uplet):
            
            imgs[i] = transforms.Resize(512)(imgs[i])
            imgs[i] = transforms.CenterCrop(512)(imgs[i])

            # appliquer fovéation + preprocess
            imgs[i] = self.preprocess(self.fovea_transform(imgs[i]))
            
        return imgs, y

    def __len__(self):
        return len(self.base_dataset)"""

class FoveatedUpletDataset(torch.utils.data.Dataset):
    """
    Pipeline complet :
      1. Resize + CenterCrop sur l'image brute  (une fois)
      2. ShiftZoomUplet → n_uplet vues PIL       (par vue)
      3. FoveatedPyramidTransform                (par vue)
      4. ToTensor + Normalize                    (par vue)
    """
    def __init__(self,
                 root,           # ImageFolder SANS transform (ou transform=None)
                 shift_zoom_uplet,
                 output_size   = 128,
                 resize        = 512,
                 crop          = 512,
                 mean          = (0.485, 0.456, 0.406),
                 std           = (0.229, 0.224, 0.225)):
        self.base    = root
        self.sz      = shift_zoom_uplet

        # ── pré-traitement image brute (une fois par sample) ────────────
        self.pre = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
        ])

        # ── post-traitement par vue ──────────────────────────────────────
        self.fov       = FoveatedPyramidTransform(output_size=output_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __getitem__(self, idx):
        img, label = self.base[idx]          # img = PIL brute

        # 1. Resize / crop (une seule fois)
        img = self.pre(img)                  # PIL 512×512

        # 2. n_uplet vues shiftées/zoomées
        views = self.sz(img)                 # liste de (pil, sx, sy, zoom)

        mosaics, sxs, sys_, zooms = [], [], [], []
        for pil, sx, sy, zoom in views:

            # 3. Mosaïque fovéale
            mosaic = self.fov(pil)           # PIL 128×128

            # 4. ToTensor + Normalize
            mosaic = self.to_tensor(mosaic)  # (3, 128, 128)  float32 [0,1]
            mosaic = self.normalize(mosaic)  # normalisé ImageNet

            mosaics.append(mosaic)
            sxs.append(sx)
            sys_.append(sy)
            zooms.append(zoom)

        return (
            torch.stack(mosaics),                          # (V, 3, 128, 128)
            torch.tensor(sxs,   dtype=torch.float32),      # (V,)
            torch.tensor(sys_,  dtype=torch.float32),      # (V,)
            torch.tensor(zooms, dtype=torch.float32),      # (V,)
            label,
        )

    def __len__(self):
        return len(self.base)



def make_dataloader(root, shiftzoom_transform, fovea_transform, batch_size=32, num_workers=4, limit=None): # limit=500 pour un test rapide
    dataset = FoveatedUpletDataset(
        root=root,
        shiftzoom_transform=shiftzoom_transform,
        fovea_transform=fovea_transform,
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)