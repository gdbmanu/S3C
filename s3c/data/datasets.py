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
    

class FoveatedUpletDataset(Dataset):
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
        return len(self.base_dataset)


def make_dataloader(root, shiftzoom_transform, fovea_transform, batch_size=32, num_workers=4, limit=None): # limit=500 pour un test rapide
    dataset = FoveatedUpletDataset(
        root=root,
        shiftzoom_transform=shiftzoom_transform,
        fovea_transform=fovea_transform,
        limit=limit
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)