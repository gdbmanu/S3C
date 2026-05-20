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

    def forward(self, X):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)        # (B, k, d_model)
        return self.mab(S, X)               # (B, k, d_model)
    
class SAB(nn.Module):
    """Set Attention Block — full O(n²) attention, fine for small n"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(self, X):
        return self.mab(X, X)  # self-attention sur l'ensemble


class FovealSetTransformer(nn.Module):
    def __init__(self, input_dim=768, 
                 n_heads=8, n_sab=2, n_classes=1000, dropout=0.1, predict=True, jepa_heads=False, proj_dim=256):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            SAB(input_dim, n_heads, dropout) for _ in range(n_sab)
        ])
        self.pma = PMA(input_dim, n_heads, k=1, dropout=dropout)
        self.predict = predict
        if predict:
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, n_classes),
            )
        self.jepa_heads = jepa_heads
        if jepa_heads:
            jepa_dim = proj_dim
            # Tête JEPA — projection vers espace de prédiction
            # MLP léger, pas besoin de grande capacité
            self.jepa_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, jepa_dim),
            )
            sigreg_dim = proj_dim
            # Tête SIGReg — projection vers haute dimension comme DINO
            # MLP 3 couches avec GELU, même structure que DINO
            self.sigreg_head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, sigreg_dim),
                # Pas de normalisation finale — SIGReg doit voir
                # les embeddings bruts pour enforcer N(0, I)
            )            

    def forward(self, x):
        # X: (B, n, 768), n entre 2 et 15
        for sab in self.encoder:
            x = sab(x)
        x = self.pma(x).squeeze(1)
        if self.jepa_heads:
            return x, self.jepa_head(x), self.sigreg_head(x)
        else:
            if self.predict:
                return self.head(x)
            else:
                return x
