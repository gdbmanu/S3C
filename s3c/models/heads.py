import torch
import torch.nn as nn


class ShiftPredictor(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # sortie (dx, dy)
        )

    def forward(self, z1, z2):
        x = torch.cat([z1, z2], dim=-1)
        return self.net(x)
    
class DualPredictor(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, 2 * emb_dim),
            nn.ReLU(),
        )

        self.shift_head = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # sortie (dx, dy)
        )

        self.label_head = nn.Linear(2 * emb_dim, 1000)
            
    def forward(self, z1, z2):
        x = torch.cat([z1, z2], dim=-1)
        x = self.net(x)
        return self.shift_head(x), self.label_head(x)
    
class TriplePredictor(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, 2 * emb_dim),
            nn.ReLU(),
        )

        self.shift_head = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # sortie (dx, dy)
        )

        self.norm = nn.LayerNorm(2 * emb_dim)
        self.label_head = nn.Linear(2 * emb_dim, 1000)
            
    def forward(self, z1, z2, z3):
        x1 = torch.cat([z1, z2], dim=-1)
        x1 = self.net(x1)
        x2 = torch.cat([z2, z3], dim=-1)
        x2 = self.net(x2)
        x3 = torch.cat([z3, z1], dim=-1)
        x3 = self.net(x3)
        x_sum = x1 + x2 + x3
        return self.shift_head(x1), self.shift_head(x2), self.shift_head(x3), self.label_head(self.norm(x_sum))
    
class TrianglePredictor(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * emb_dim, 6 * emb_dim),
            nn.ReLU(),
            nn.Linear(6 * emb_dim, 3 * emb_dim),
            nn.ReLU(),
        )

        self.triangle_head = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # sortie (dx, dy)
        )

        self.norm = nn.LayerNorm(3 * emb_dim)
        self.label_head = nn.Linear(3 * emb_dim, 1000)
            
    def forward(self, z1, z2, z3):
        x = torch.cat([z1, z2, z3], dim=-1)
        x = self.net(x)
        return self.triangle_head(x), self.label_head(self.norm(x))
    

class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, 
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, Q, K):
        # Q attends to K
        attn_out, _ = self.attn(Q, K, K)
        Q = self.norm1(Q + attn_out)
        Q = self.norm2(Q + self.ffn(Q))
        return Q
    
class PMA(nn.Module):
    """Pooling by Multihead Attention — k seed vectors"""
    def __init__(self, d_model, n_heads, k=1, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d_model))
        self.mab = MAB(d_model, n_heads, dropout)
        self.sab = SAB(d_model, n_heads, dropout)

    def forward(self, X):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)        # (B, k, d_model)
        #return self.mab(S, X)              # (B, k, d_model)
        return self.sab(self.mab(S, X))     # (B, k, d_model)
    
class SAB(nn.Module):
    """Set Attention Block — full O(n²) attention, fine for small n"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(self, X):
        return self.mab(X, X)  # self-attention sur l'ensemble


class FovealSetTransformer(nn.Module):
    def __init__(self, input_dim=768, 
                 n_heads=8, n_sab=2, k=1, n_classes=1000, dropout=0.1, predict=True, proj=False, proj_dim=256):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            SAB(input_dim, n_heads, dropout) for _ in range(n_sab)
        ])
        self.pma = PMA(input_dim, n_heads, k=k, dropout=dropout)
        self.k = k
        self.predict = predict
        if predict:
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim * k),
                nn.Linear(input_dim * k, n_classes),
            )
        self.proj = proj
        self.proj_dim = proj_dim
        if proj:
            # MLP 3 couches avec GELU, même structure que DINO
            self.proj_head = nn.Sequential(
                #nn.Linear(input_dim, input_dim),
                #nn.GELU(),
                #nn.Linear(input_dim, input_dim),
                #nn.GELU(),
                nn.Linear(input_dim, proj_dim),
                # Pas de normalisation finale — SIGReg doit voir
                # les embeddings bruts pour enforcer N(0, I)
            )            

    def forward(self, x):
        # X: (B, n, 768), n entre 2 et 15
        for sab in self.encoder:
            x = sab(x)
        if self.k==1:
            x = self.pma(x).squeeze(1)
            if self.proj:
                return self.proj(x)
            else:
                if self.predict:
                    return self.head(x)
                else:
                    return x
        else:
            x = self.pma(x)
            B, k, emb_dim = x.shape
            if self.proj:
                x = x.view(B * k, emb_dim)
                x = self.proj_head(x)
                return x.view(B, k, self.proj_dim)
            else:
                if self.predict:
                    return self.head(x.view(B, k * emb_dim)) 
                else:
                    return x
                
class AttentionPooling(nn.Module):
    """Gated attention pooling à la ABMIL (Ilse et al. 2018)"""
    def __init__(self, d_model, hidden=256, inv_temp = 1.):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.z_norm = nn.LayerNorm(d_model)
        self.inv_temp = inv_temp

    def forward(self, z):
        # z: (B, n, d)
        # z = self.z_norm(z)
        w = self.attn(self.z_norm(z))          # (B, n, 1)
        w = torch.softmax(w, dim=1)
        return (w * z * self.inv_temp).sum(dim=1), w # (B, d)
                

class SeedBlock(nn.Module):
    """
    Un bloc = 
      - self-attention sur les vues (les vues se transforment entre elles)
      - cross-attention seeds → vues (seeds lisent les vues transformées)
      - self-attention sur les seeds (seeds se coordonnent)
    """
    def __init__(self, d_model, n_heads, dropout=0.1, self_att=False):
        super().__init__()

        # Self-attention sur les vues
        self.view_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.view_norm1 = nn.LayerNorm(d_model)
        self.view_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.view_norm2 = nn.LayerNorm(d_model)

        # Cross-attention : seeds lisent les vues
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.seed_norm1 = nn.LayerNorm(d_model)

        # Self-attention sur les seeds
        self.self_att = self_att
        if self_att:
            self.seed_self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.seed_norm2 = nn.LayerNorm(d_model)

        self.seed_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.seed_norm3 = nn.LayerNorm(d_model)


    def forward(self, seeds, views):
        # ── 1. Vues se transforment entre elles ──────────────────────

        v = self.view_norm1(views)
        h_views, _ = self.view_self_attn(v, v, v)
        views = views + h_views
        v2 = self.view_norm2(views)
        views = views + self.view_ffn(v2)

        # ── 2. Seeds lisent les vues transformées ─────────────────────
        s = self.seed_norm1(seeds)
        h_seeds, _ = self.cross_attn(s, v2, v2)
        seeds = seeds + h_seeds 

        # ── 3. Seeds se coordonnent ───────────────────────────────────
        if self.self_att:
            s2 = self.seed_norm2(seeds)
            h_self, _ = self.seed_self_attn(s2, s2, s2)
            seeds = seeds + h_self
            seeds = seeds + self.seed_ffn(self.seed_norm3(seeds))
        else:
            seeds = seeds + self.seed_ffn(self.seed_norm3(seeds))

        return seeds, views   # les deux évoluent


class IterativeSeedTransformer(nn.Module):
    def __init__(self, input_dim=768, d_model=768,
                 n_heads=12, n_seeds=4, n_blocks=4, dropout=0.1, self_att=False, normalize=False):
        super().__init__()
        self.proj  = (nn.Linear(input_dim, d_model)
                      if input_dim != d_model else nn.Identity())
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.blocks = nn.ModuleList([
            SeedBlock(d_model, n_heads, dropout, self_att) for _ in range(n_blocks)
        ])
        self.norm_seeds = nn.LayerNorm(d_model)
        self.normalize = normalize
        #self.norm_views = nn.LayerNorm(d_model)

    def forward(self, X):
        B = X.size(0)
        views = self.proj(X)
        seeds = self.seeds.expand(B, -1, -1).clone()

        for block in self.blocks:
            seeds, views = block(seeds, views)   # co-évolution

        if self.normalize:
            return self.norm_seeds(seeds)   # (B, n_seeds, d_model)
        else:
            return seeds
        # views finales disponibles si besoin : self.norm_views(views)