import torch
import torch.nn as nn
import torch.nn.functional as F


class FoveatedMultiViT(nn.Module):
    def __init__(self, model, norm=True):
        super().__init__()
        self.model = model
        self.model.pos_embed.requires_grad_(False)
        self.norm = norm

    def forward_single(self, x):
        """x : (B, 3, H, W) → (B, 1+N, D)"""
        B   = x.shape[0]
        x   = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.model.pos_embed
        x   = self.model.pos_drop(x)
        x   = self.model.blocks(x)
        x   = self.model.norm(x)
        return x                                        # (B, 1+N, D)

    def forward_multi(self, views):
        """
        mosaics : (B, V, 3, H, W)
        Retourne : (B, V, 1+N, D)
          chaque vue est traitée indépendamment
          → un CLS token par vue
        """
        B, V, C, H, W = views.shape

        # ── agréger V sur la dim batch → traitement indépendant par vue ──
        x = views.view(B * V, C, H, W)           # (B*V, 3, H, W)

        # forward complet — identique à forward_single
        x = self.model.patch_embed(x)               # (B*V, N, D)
        N = x.shape[1]
        D = x.shape[2]

        cls = self.model.cls_token.expand(B*V, -1, -1)  # (B*V, 1, D)
        x   = torch.cat([cls, x], dim=1)            # (B*V, 1+N, D)
        x   = x + self.model.pos_embed              # (B*V, 1+N, D)
        x   = self.model.pos_drop(x)
        x   = self.model.blocks(x)
        if self.norm:
            x   = self.model.norm(x)                    # (B*V, 1+N, D)

        # ── reshape pour retrouver la dimension vue ───────────────────────
        x = x.view(B, V, 1 + N, D)
        return x[:, :, 0, :]                        # (B, V, D) CLS tokens only
       
    
    def forward_multi_embeddings(self, views): # old version (to be recycled)
        B, V, C, H, W = views.shape
        D = self.model.embed_dim

        # 1. Patch embedding
        p_all = self.model.patch_embed(
            views.view(B * V, C, H, W)
        ).view(B, V, -1, D)                                      # (B, V, N, D)
        N = p_all.shape[2]

        # 2. Pos embed — séparer CLS pos et patches pos
        pos_cls     = self.model.pos_embed[:, :1, :]             # (1, 1, D)
        pos_patches = self.model.pos_embed[:, 1:, :]             # (1, N, D)

        # 3. Même pos embed pour toutes les vues
        # expand : (1, N, D) → (B, V, N, D)
        pos_patches_exp = pos_patches.unsqueeze(1).expand(B, V, N, D)
        p_all = p_all + pos_patches_exp                          # (B, V, N, D)

        # 4. CLS token + son pos embed (un seul, partagé)
        cls = self.model.cls_token.expand(B, -1, -1) + pos_cls  # (B, 1, D)

        # 5. Concaténer
        p_flat = p_all.contiguous().reshape(B, V * N, D)         # (B, V*N, D)
        x      = torch.cat([cls, p_flat], dim=1)                 # (B, 1+V*N, D)

        # 6. Transformer
        x = self.model.pos_drop(x)
        x = self.model.blocks(x)
        if self.norm:
            x = self.model.norm(x)
        return x                                                  # (B, 1+V*N, D)

    def forward(self, views, *args, **kwargs):
        if views.dim() == 4:
            return self.forward_single(views)
        return self.forward_multi(views)
    
def build_foveated_pos_embed(model,
                            base_img_size=224,
                            patch_size=8,
                            mosaic_sub_patch_grid=8,
                            scales=(1.0, 0.5, 0.25, 0.125),
                            device=None):
    """
    Reconstruit model.pos_embed pour une mosaïque fovéale 2x2.
    - model: timm ViT (avec model.pos_embed shape (1, 1+H*W, D))
    - base_img_size: taille sur laquelle le pos_embed d'origine a été appris (ex. 224)
    - patch_size: taille du patch (ex. 8)
    - mosaic_sub_patch_grid: nombre de patches par sous-image (ici 8 -> 64px / 8 = 8)
    - scales: 4 échelles centrées à extraire (1.0, 0.5, 0.25, 0.125)
    - device: device cible (None -> prend device des poids)
    Retour:
      remplace model.pos_embed et renvoie le nouveau pos_embed (torch.nn.Parameter)
    """
    if device is None:
        device = next(model.parameters()).device

    # shape initiale
    pos = model.pos_embed.detach().to(device)   # (1, 1 + N_old, D)
    cls_token = pos[:, :1, :].clone()          # (1,1,D)
    patch_pos = pos[:, 1:, :].clone()          # (1, N_old, D)

    # grille d'origine (en patches)
    grid_old = int((base_img_size // patch_size))
    assert grid_old * grid_old == patch_pos.shape[1], "Incompatible pos_embed size vs base_img_size/patch_size"

    D = patch_pos.shape[-1]

    # reshape en (1, D, H_old, W_old)
    patch_pos_map = patch_pos.reshape(1, grid_old, grid_old, D).permute(0, 3, 1, 2)  # (1, D, H_old, W_old)

    # For each scale: crop central square in patch-grid coordinates, then interpolate to (mosaic_sub_patch_grid, mosaic_sub_patch_grid)
    sub_maps = []
    for s in scales:
        # nombre de patches du côté à extraire dans la grille d'origine
        side = max(1, int(round(grid_old * s)))
        # centre
        start = (grid_old - side) // 2
        end = start + side
        # extraire la région centrale (en index patch)
        cropped = patch_pos_map[:, :, start:end, start:end]  # (1, D, side, side)
        # si side==1 on a shape (1,D,1,1) -> ok pour interpolation
        # interpoler vers (mosaic_sub_patch_grid, mosaic_sub_patch_grid)
        interp = F.interpolate(cropped, size=(mosaic_sub_patch_grid, mosaic_sub_patch_grid),
                               mode='bicubic', align_corners=False)
        # stocker
        sub_maps.append(interp)  # each is (1, D, sub_h, sub_w) where sub_h=sub_w=mosaic_sub_patch_grid

    # On a 4 sous-cartes (1,D,8,8) chacune (si mosaic_sub_patch_grid=8)
    # On assemble en mosaïque 2x2 : ordre choisi = [global, interm, peri, foveal]
    # top row: sub0 | sub1 ; bottom row: sub2 | sub3
    top_row = torch.cat([sub_maps[0], sub_maps[1]], dim=3)    # concat horizontalement => (1, D, 8, 16)
    bottom_row = torch.cat([sub_maps[2], sub_maps[3]], dim=3) # (1, D, 8, 16)
    full_map = torch.cat([top_row, bottom_row], dim=2)       # concat verticalement => (1, D, 16, 16)

    # transformer en (1, 16*16, D)
    Hnew, Wnew = full_map.shape[2], full_map.shape[3]
    assert Hnew == Wnew, "La mosaïque doit être carrée"
    new_num_patches = Hnew * Wnew
    new_patch_pos = full_map.permute(0, 2, 3, 1).reshape(1, new_num_patches, D)  # (1, newN, D)

    # concat CLS token
    new_pos_embed = torch.cat([cls_token.to(device), new_patch_pos], dim=1)  # (1, 1+newN, D)

    # Remplacer dans le modèle
    model.pos_embed = torch.nn.Parameter(new_pos_embed)

    return model.pos_embed
