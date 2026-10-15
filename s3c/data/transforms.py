import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms

from torchvision.transforms import functional as FF

from PIL import Image



class ShiftZoomUplet:
    def __init__(self, zoom=1.5, std=0.3, n_uplet=5, start_center=False):
        self.zoom    = zoom
        self.std     = std
        self.n_uplet = n_uplet
        self.start_center = start_center
        assert self.zoom >= 1.0, "zoom < 1 interdit : le crop dépasserait l'image originale"

    def shift_zoom(self, img, x, y):
        w, h = img.size
        size_ref = min(w, h)
        w_zoomed   = int(w * self.zoom)
        h_zoomed   = int(h * self.zoom)
        zoomed_ref = min(w_zoomed, h_zoomed)   # petit côté zoomé — référence du shift

        img_zoomed = FF.resize(img, (h_zoomed, w_zoomed),
                            interpolation=Image.BICUBIC)
        y_prim = y

        # centre de fixation dans l'espace zoomé
        # x,y ∈ [-1,1] définis par rapport au petit côté original
        cx = w_zoomed / 2 + zoomed_ref * x / 2 #zoomed_ref / 2 * (1 + x / 2)   # ∈ [zoomed_ref/4, 3*zoomed_ref/4]
        cy = h_zoomed / 2 + zoomed_ref * y_prim / 2 #zoomed_ref / 2 * (1 + y / 2)

        # crop de taille originale (w, h) centré en (cx, cy)
        left   = int(cx - size_ref/2) #w / 2)
        top    = int(cy - size_ref/2) #h / 2)
        right  = left + size_ref #w
        bottom = top  + size_ref #h

        # padding reflect uniquement si on dépasse l'image zoomée réelle
        pl = max(0, -left);             
        pt = max(0, -top)
        pr = max(0, right  - w_zoomed); 
        pb = max(0, bottom - h_zoomed)

        if any([pl, pt, pr, pb]):
            img_zoomed = FF.pad(img_zoomed, (pl, pt, pr, pb),
                                padding_mode='reflect')
        left += pl
        top  += pt

        return img_zoomed.crop((left, top, left + size_ref, top + size_ref)) #left + w, top + h))

    def __call__(self, img, views_list = None):
        """
        Retourne une liste de n_uplet tuples :
          (img_shifted: PIL.Image, sx: float, sy: float, zoom: float)
        """
        views = []
        if views_list is None:
            for i in range(self.n_uplet):
                if self.start_center and i == 0:
                    sx, sy = [0, 0]
                else:
                    sx, sy = np.random.normal(0, self.std, 2)
                    sx, sy = float(np.clip(sx, -1, 1)), float(np.clip(sy, -1, 1))
                img_shifted = self.shift_zoom(img, sx, sy)
                views.append((img_shifted, sx, sy, self.zoom)) 
        else:
            for (sx, sy) in views_list:
                img_shifted = self.shift_zoom(img, sx, sy)
                views.append((img_shifted, sx, sy, self.zoom))

        return views



class ShiftZoomGrid(ShiftZoomUplet):
    def __init__(self, zoom=1.5, n_grid=11):
        # n_uplet = n_grid² mais on n'utilise pas le tirage aléatoire
        super().__init__(zoom=zoom, std=0.0, n_uplet=n_grid * n_grid)
        self.n_grid = n_grid
        # Grille fixe de shifts entre -0.75 et 0.75
        self.grid = torch.linspace(-0.75, 0.75, n_grid)

    def __call__(self, img):
        views = []
        for y in self.grid:
            for x in self.grid:
                sx, sy = float(x), float(y)
                img_shifted = self.shift_zoom(img, sx, sy)
                views.append((img_shifted, sx, sy, self.zoom))
        return views
    
class FoveatedPyramidTransform:
    def __init__(self, output_size=128, to_torch=False):
        """
        output_size : taille finale de la mosaïque (typiquement 128)
        Chaque sous-image fait output_size // 2 (ex: 64x64 si output_size=128)
        """
        self.output_size = output_size
        self.resized_size = output_size // 2
        self.to_torch = to_torch

    def __call__(self, img, level=3):
        """
        img : PIL.Image ou Tensor [C,H,W]
        Retourne : Tensor [3, output_size, output_size]
        """
        if isinstance(img, torch.Tensor):
            img = FF.to_pil_image(img)

        w, h = img.size
        scales = [1.0, 0.5, 0.25, 0.125]
        levels = [3, 2, 1, 0]
        crops = []


        for s, l in zip(scales, levels):
            # Taille du crop à cette échelle (en px, pour chaque dimension)
            w_side = int(w * s)
            h_side = int(h * s)

            # Coordonnées centrées
            left = (w - w_side) // 2
            top = (h - h_side) // 2
            right = left + w_side
            bottom = top + h_side

            # Crop + resize
            crop = img.crop((left, top, right, bottom))
            crop_resized = FF.resize(crop, (self.resized_size, self.resized_size), interpolation=FF.InterpolationMode.BICUBIC)
            if level < l:
                crop_resized = np.zeros_like(crop_resized)
            crops.append(FF.to_tensor(crop_resized))

        # Mosaïque 2×2 :
        # [ global | intermédiaire ]
        # [ péri-fovéal | fovéal ]
        top_row = torch.cat([crops[0], crops[1]], dim=2)    # concat horizontalement (largeur)
        bottom_row = torch.cat([crops[2], crops[3]], dim=2)
        mosaic = torch.cat([top_row, bottom_row], dim=1)    # concat verticalement (hauteur)

        # S’assurer que la taille finale est bien 128×128
        mosaic = FF.resize(FF.to_pil_image(mosaic), (self.output_size, self.output_size), interpolation=FF.InterpolationMode.BICUBIC)
        #mosaic = FF.to_tensor(mosaic)
        if self.to_torch:
            mosaic = FF.to_tensor(mosaic)

        return mosaic

class FlatTransform:
    """
    Équivalent 'flat' de FoveatedPyramidTransform.
    level (0..n_levels-1) : plus il est haut, plus la zone centrale conservée est grande.
    Hors de ce carré central, les pixels sont à zéro (noir, avant normalisation).
    level=None -> image entière (comme level=3 dans la version fovéale).
    """
    def __init__(self, output_size=128, keep_fractions=(0.5, 0.707, 0.866, 1.0)): #(0.25, 0.5, 0.75, 1.0)):
        self.output_size = output_size
        self.keep_fractions = keep_fractions       # indexé par level
        self.n_levels = len(keep_fractions)
        self.base = transforms.Compose([
            transforms.Resize(output_size),
            transforms.CenterCrop(output_size),
        ])

    def __call__(self, img, level=None):
        img = self.base(img)                       # PIL, output_size x output_size
        if level is None:
            level = self.n_levels - 1
        frac = self.keep_fractions[level]
        if frac >= 1.0:
            return img

        size = self.output_size
        side = max(1, round(size * frac))
        left = (size - side) // 2                  # même centrage que le crop fovéal
        out = Image.new(img.mode, img.size)        # noir par défaut
        out.paste(img.crop((left, left, left + side, left + side)), (left, left))
        return out