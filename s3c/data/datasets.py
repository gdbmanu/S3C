import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pycocotools.coco import COCO


from s3c.data.transforms import ShiftZoomUplet, ShiftZoomGrid, FoveatedPyramidTransform, FlatTransform

import numpy as np

import random

from pathlib import Path

from PIL import Image


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

class FoveatedCocoUpletDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, ann_file, zoom, std, n_uplet=30,
             max_labeled=5, output_size=128, min_box=4,
             dedup_labels=False, clip_labeled=False, path=False):
        self.coco = COCO(ann_file)
        self.img_root = Path(img_root)
        self.ids = sorted(self.coco.getImgIds())
        self.n_uplet, self.std = n_uplet, std
        self.max_labeled, self.min_box = max_labeled, min_box
        self.dedup_labels, self.clip_labeled = dedup_labels, clip_labeled
        self.path = path

        cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {c: i for i, c in enumerate(cat_ids)}
        self.idx2name = {i: self.coco.loadCats(c)[0]["name"]
                        for c, i in self.cat2idx.items()}

        self.shiftzoom_transform = ShiftZoomUplet(zoom=zoom, std=std, n_uplet=n_uplet)
        self.fovea_transform = FoveatedPyramidTransform(output_size=output_size)
        self.post_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
    def _labeled_targets(self, img_id, W, H):
        rng = random.Random(img_id)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
        targets = []
        for a in anns:
            x, y, w, h = a["bbox"]
            # bornage à l'image (certaines bboxes COCO dépassent légèrement)
            x0, y0 = max(0., x), max(0., y)
            x1, y1 = min(float(W), x + w), min(float(H), y + h)
            if (x1 - x0) < self.min_box or (y1 - y0) < self.min_box:
                continue
            targets.append(((x0 + x1) / 2, (y0 + y1) / 2,
                            self.cat2idx[a["category_id"]]))

        if self.dedup_labels:
            by_cls = {}
            for t in targets:
                by_cls.setdefault(t[2], []).append(t)
            targets = [rng.choice(v) for v in by_cls.values()]

        if len(targets) > self.max_labeled:
            targets = rng.sample(targets, self.max_labeled)
        return targets
    
    def  __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.img_root / info["file_name"]).convert("RGB")
        W, H = img.size
        ref = min(W, H)

        # 1) vues avec label : centrées sur les bboxes
        shifts, labels = [], []
        for bx, by, lab in self._labeled_targets(img_id, W, H):
            sx = 2 * (bx - W / 2) / ref
            sy = 2 * (by - H / 2) / ref
            if self.clip_labeled:
                sx, sy = max(-1., min(1., sx)), max(-1., min(1., sy))
            shifts.append((sx, sy))
            labels.append(lab)

        # 2) vues sans label : positions aléatoires, label -1
        n_rand = self.n_uplet - len(shifts)
        for _ in range(n_rand):
            sx, sy = np.random.normal(0, self.std, 2)
            shifts.append((float(np.clip(sx, -1, 1)), float(np.clip(sy, -1, 1))))
            labels.append(-1)

        # 3) un seul appel : crops pour toute la liste de décalages
        views = self.shiftzoom_transform(img, views_list=shifts)

        fov, sxs, sys_, zooms = [], [], [], []
        for view, sx, sy, z in views:
            view = transforms.Resize(512)(view)
            view = transforms.CenterCrop(512)(view)
            fov.append(self.post_process(self.fovea_transform(view)))
            sxs.append(sx); sys_.append(sy); zooms.append(z)

        out = (torch.stack(fov),
            torch.tensor(sxs, dtype=torch.float32),
            torch.tensor(sys_, dtype=torch.float32),
            torch.tensor(zooms, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))
        if self.path:
            out += (info["file_name"],)
        return out

    def __len__(self):
        return len(self.ids)

    
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
        d = torch.load(path, map_location="cpu") #.float()  # (n_sac, D)
        zs = d["zs"].float()
        xs = d["xs"].float()
        ys = d["ys"].float()
        return zs, xs, ys, label

    def __len__(self):
        return len(self.samples)


class CocoZDataset(Dataset):
    """
    Charge les embeddings pré-calculés depuis <root>/ (ex. coco_z/val/).
    Retourne (zs, xs, ys, labels) :
      zs     : (n_sac, D)  float32
      xs, ys : (n_sac,)    float32
      labels : (n_sac,)    long, indice de classe 0..79, ou -1 (vue aléatoire)
    Avec return_id=True, retourne en plus l'image_id COCO (int).
    """
    def __init__(self, root, return_id=False):
        self.root = Path(root).expanduser()
        self.files = sorted(self.root.glob("*.pt"))   # tri → ordre déterministe
        if not self.files:
            raise FileNotFoundError(f"Aucun .pt trouvé dans {self.root}")
        self.return_id = return_id

    def __getitem__(self, idx):
        path = self.files[idx]
        d = torch.load(path, map_location="cpu")
        zs = d["zs"].float()
        xs = d["xs"].float()
        ys = d["ys"].float()
        labels = d["labels"].long()
        if self.return_id:
            return zs, xs, ys, labels, int(path.stem)   # "000000397133" -> 397133
        return zs, xs, ys, labels

    def __len__(self):
        return len(self.files)



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
