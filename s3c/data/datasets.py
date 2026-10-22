import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from s3c.data.transforms import ShiftZoomUplet, ShiftZoomGrid, FoveatedPyramidTransform, FlatTransform

import numpy as np

import random

from pathlib import Path



from pathlib import Path
import torch
from torchvision import transforms

class FoveatedUpletDataset(torch.utils.data.Dataset):

    def __init__(self,
                 base_folder,            # ImageFolder
                 zoom,
                 std,
                 n_uplet,
                 transform="fovea",      # "fovea" ou "flat"
                 output_size=128,
                 n_levels=4,
                 start_center=False,
                 limit=None,
                 path=False,
                 random_levels=True, 
                 fixed_level=None):                 

        self.base = datasets.ImageFolder(base_folder, transform=None)
        self.shiftzoom_transform = ShiftZoomUplet(zoom=zoom, std=std,
                                                  n_uplet=n_uplet,
                                                  start_center=start_center)

        if transform == "fovea":
            self.fovea_transform = FoveatedPyramidTransform(output_size=output_size)
        elif transform == "flat":
            self.fovea_transform = FlatTransform(output_size)
        else:
            raise ValueError(f"transform doit valoir 'fovea' ou 'flat', reçu : {transform!r}")
        
        self.random_levels = random_levels
        self.fixed_level = fixed_level
        if fixed_level is not None and not 0 <= fixed_level < self.n_levels:
            raise ValueError(f"fixed_level doit être dans [0, {self.n_levels - 1}], reçu : {fixed_level}")

        self.post_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        if limit:
            self.base.samples = self.base.samples[:limit]

        self.path = path
        if path:
            self.path_root = Path(base_folder.root)

    def __getitem__(self, idx):
        img, label = self.base[idx]                      # PIL brute
        views = self.shiftzoom_transform(img)            # liste de (view, sx, sy, zoom)

        fov_views, sxs, sys_, zooms = [], [], [], []
        for view, sx, sy, zoom in views:
            if self.random_levels:
                fov_views.append(self.post_process(self.fovea_transform(view, level=np.random.randint(self.n_levels))))
            elif self.fixed_level is not None:
                fov_views.append(self.post_process(self.fovea_transform(view, level=self.fixed_level)))
            else:
                fov_views.append(self.post_process(self.fovea_transform(view)))
            sxs.append(sx)
            sys_.append(sy)
            zooms.append(zoom)

            # appliquer fovéation + post_process

        out = (
            torch.stack(fov_views),                        # (V, 3, output_size, output_size)
            torch.tensor(sxs,   dtype=torch.float32),      # (V,)
            torch.tensor(sys_,  dtype=torch.float32),      # (V,)
            torch.tensor(zooms, dtype=torch.float32),      # (V,)
            label,
        )
        if self.path:
            path_info, _ = self.base.samples[idx]
            out += (str(Path(path_info).relative_to(self.path_root)),)
        return out

    def __len__(self):
        return len(self.base)

class FoveatedGridDataset(FoveatedUpletDataset):
    def __init__(self,
                 base_folder,
                 zoom,
                 n_grid      = 11,
                 transform   = "fovea",     # "fovea" ou "flat"
                 output_size = 128,
                 n_levels=4,
                 start_center=True, 
                 limit=None, 
                 random_levels=True, 
                 fixed_level=None,
                 path        = False):
        
        super().__init__(
            base_folder  = base_folder,
            zoom         = zoom,
            std          = 0.0,
            n_uplet      = n_grid * n_grid,
            transform    = transform,
            n_levels     = n_levels,
            output_size  = output_size,
            start_center = start_center,
            limit        = limit,
            random_levels= random_levels,
            fixed_level  = fixed_level,
            path         = path,
        )
        # Remplace le ShiftZoomUplet aléatoire par la grille déterministe
        self.shiftzoom_transform = ShiftZoomGrid(zoom=zoom, n_grid=n_grid)
    
class FoveatedPairDataset(Dataset):
    def __init__(self, root, zoom, std, transform="fovea", output_size=128,
                 n_levels=4,
                 start_center=True, limit=None, random_levels=True, fixed_level=None):
        self.base_dataset = datasets.ImageFolder(root, transform=None)
        self.shiftzoom_transform = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=2,
                                                  start_center=start_center)

        if transform == "fovea":
            self.fovea_transform = FoveatedPyramidTransform(output_size=output_size)
        elif transform == "flat":
            self.fovea_transform = FlatTransform(output_size)
        else:
            raise ValueError(f"transform doit valoir 'fovea' ou 'flat', reçu : {transform!r}")
        
        self.n_levels = getattr(self.fovea_transform, "n_levels", n_levels)

        self.random_levels = random_levels
        self.fixed_level = fixed_level
        if fixed_level is not None and not 0 <= fixed_level < self.n_levels:
            raise ValueError(f"fixed_level doit être dans [0, {self.n_levels - 1}], reçu : {fixed_level}")

        self.post_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        if limit:
            self.base_dataset.samples = self.base_dataset.samples[:limit]

    @torch.no_grad()
    def __getitem__(self, idx):          
        max_retries = 10
        for attempt in range(max_retries):
            try:
                img, _ = self.base_dataset[idx]


                # 2 views only
                views = self.shiftzoom_transform(img)
                # views : (img_shifted, sx, sy, zoom)) 

                img1, x1, y1, _ = views[0]
                img2, x2, y2, _ = views[1]
                shift = torch.tensor([x2 - x1, y2 - y1], dtype=torch.float32)
            

                # appliquer fovéation + post_process
                if self.random_levels:
                    img1 = self.fovea_transform(img1, level=np.random.randint(self.n_levels))
                elif self.fixed_level is not None:
                    img1 = self.fovea_transform(img1, level=self.fixed_level)
                else:
                    img1 = self.fovea_transform(img1)    # niveau par défaut = information complète
                img1 = self.post_process(img1)
                img2 = self.fovea_transform(img2)
                img2 = self.post_process(img2)

                return img1, img2, shift

            except (OSError, IOError) as e:
                print(f"[Dataset] Erreur accès idx={idx}, tentative {attempt+1}/{max_retries} : {e}")
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(f"Impossible de charger un exemple après {max_retries} tentatives.")
    
    def __len__(self):
        return len(self.base_dataset)

    
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



def collate_uplet(batch):
    """
    Collate qui empile correctement les uplets.
    Retourne :
      mosaics : (B, n_uplet, 3, H, W)
      sxs     : (B, n_uplet)
      sys_    : (B, n_uplet)
      zooms   : (B, n_uplet)
      labels  : (B,)
    """
    mosaics = torch.stack([b[0] for b in batch])
    sxs     = torch.stack([b[1] for b in batch])
    sys_    = torch.stack([b[2] for b in batch])
    zooms   = torch.stack([b[3] for b in batch])
    labels  = torch.tensor([b[4] for b in batch])
    return mosaics, sxs, sys_, zooms, labels


# %%
def make_dataloader(root, std, n_views, zoom=1.5, start_center=False, batch_size=32, num_workers=4, limit=None, 
                    transform="fovea",resolution=128, n_levels=4,
                         random_levels=True, fixed_level=None): # limit=500 pour un test rapide
    dataset = FoveatedUpletDataset(
        root,
        zoom=zoom,
        std=std,
        transform=transform, 
        output_size=resolution,
        n_levels=n_levels,
        random_levels=random_levels, 
        fixed_level=fixed_level,
        start_center=start_center, 
        n_uplet=n_views,
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



def make_pair_dataloader(root, std, zoom=1.5, start_center=True, batch_size=32,
                         num_workers=4, limit=None, transform="fovea",
                         resolution=128, n_levels=4,
                         random_levels=True, fixed_level=None):
    dataset = FoveatedPairDataset(
        root=root, zoom=zoom, std=std,
        transform=transform, output_size=resolution,
        n_levels=n_levels,
        random_levels=random_levels, fixed_level=fixed_level,
        start_center=start_center, limit=limit,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers)
