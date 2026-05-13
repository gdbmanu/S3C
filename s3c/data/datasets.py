import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from s3c.data.transforms import ShiftZoomUplet, FoveatedPyramidTransform

import numpy as np

import random

from pathlib import Path



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
   
    def __init__(self,
                 base_folder,           # ImageFolder 
                 zoom, 
                 std,
                 n_uplet,
                 output_size   = 128,
                 start_center=False, 
                 limit=None,
                 path=False
                 ):
        
        self.base    = base_folder
        self.shiftzoom_transform      = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=n_uplet, start_center=start_center)

        # ── pré-traitement image brute (une fois par sample) ────────────
        #self.pre = transforms.Compose([
        #    transforms.Resize(resize),
        #    transforms.CenterCrop(crop),
        #])

        # ── post-traitement par vue ──────────────────────────────────────
        self.fovea_transform       = FoveatedPyramidTransform(output_size=output_size)

        self.post_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

        if limit:
            self.base_dataset.samples = self.base_dataset.samples[:limit]

        self.path = path
        if path:
            self.path_root = Path(base_folder.root)


    def __getitem__(self, idx):
        img, label = self.base[idx]          # img = PIL brute
        if self.path:
            path_info, _ = self.base.samples[idx]
            #print('Path info:', path_info)
            rel_path    = Path(path_info).relative_to(self.path_root)

        # 1. Resize / crop (une seule fois)
        # img = self.pre(img)                  # PIL 512×512

        # 2. n_uplet vues shiftées/zoomées
        views = self.shiftzoom_transform(img)                 # liste de (view, sx, sy, zoom)

        fov_views, sxs, sys_, zooms = [], [], [], []
        for view, sx, sy, zoom in views:

            view = transforms.Resize(512)(view)
            view = transforms.CenterCrop(512)(view)

            # 3. Mosaïque fovéale
            fov_view = self.fovea_transform(view)           # PIL 128×128

            # 4. ToTensor + Normalize
            fov_view = self.post_process(fov_view)

            fov_views.append(fov_view)
            sxs.append(sx)
            sys_.append(sy)
            zooms.append(zoom)
        if not self.path:
            return (
                torch.stack(fov_views),                          # (V, 3, 128, 128)
                torch.tensor(sxs,   dtype=torch.float32),      # (V,)
                torch.tensor(sys_,  dtype=torch.float32),      # (V,)
                torch.tensor(zooms, dtype=torch.float32),      # (V,)
                label,
            )
        else:
            return (
                torch.stack(fov_views),                          # (V, 3, 128, 128)
                torch.tensor(sxs,   dtype=torch.float32),      # (V,)
                torch.tensor(sys_,  dtype=torch.float32),      # (V,)
                torch.tensor(zooms, dtype=torch.float32),      # (V,)
                label,
                str(rel_path)
            )

    def __len__(self):
        return len(self.base)
    
class FoveatedPairDataset(Dataset):
    def __init__(self, 
                 root, 
                 zoom, 
                 std, 
                 output_size=128,
                 start_center=True, 
                 limit=None):
        #self.pre_process = transforms.Compose([
        #    transforms.Resize(resize),
        #    transforms.CenterCrop(crop),
        #])
        self.base_dataset = datasets.ImageFolder(root, transform=None)
        self.shiftzoom_transform = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=2, start_center=start_center)
        self.fovea_transform       = FoveatedPyramidTransform(output_size=output_size)
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
                #img = self.pre_process(img)                  # PIL 512×512

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
                img1 = self.fovea_transform(img1, level=level)
                img1 = self.post_process(img1)
                img2 = self.fovea_transform(img2)
                img2 = self.post_process(img2)

                return img1, img2, shift

            except (OSError, IOError) as e:
                print(f"[Dataset] Erreur accès idx={idx}, tentative {attempt+1}/{max_retries} : {e}")
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Impossible de charger un exemple après {max_retries} tentatives.")
    
    
class ImageNetZDataset(Dataset):
    """
    Charge les embeddings pré-calculés depuis imagenet_z/.
    Retourne (z, label) avec z de shape (n_sac, D).
    """
    def __init__(self, root):
        self.root    = Path(root)
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir()
                                if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            label = self.class_to_idx[cls]
            for pt in (self.root / cls).glob("*.pt"):
                self.samples.append((pt, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        dict = torch.load(path, map_location="cpu") #.float()  # (n_sac, D)
        zs = dict["zs"].float()
        xs = dict["xs"].float()
        ys = dict["ys"].float()
        return zs, xs, ys, label

    def __len__(self):
        return len(self.samples)


# %%
def make_dataloader(root, zoom, std, n_views, start_center=False, batch_size=32, num_workers=4, limit=None): # limit=500 pour un test rapide
    dataset = FoveatedUpletDataset(
        root=root,
        zoom=zoom,
        std=std,
        start_center=start_center, 
        n_uplet=n_views,
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



def make_pair_dataloader(root, zoom, std, start_center=True, batch_size=32, num_workers=4, limit=None): # limit=500 pour un test rapide
    dataset = FoveatedPairDataset(
        root=root,
        zoom=zoom,
        std=std,
        start_center=start_center, 
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)