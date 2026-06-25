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
    

class PosPredictor(nn.Module):
    """
    Prédit la position (x, y) d'une vue z à partir des k seeds
    et de l'embedding DINO de cette vue.

    Poids et LayerNorm DISTINCTS par seed.

    s : (B, k, emb_dim)  — seeds
    z : (B, emb_dim)     — embedding DINO de la vue à localiser
    """
    def __init__(self, emb_dim=768, k=1, hidden_dim=512, out_dim=2):
        super().__init__()
        self.k = k

        # LayerNorm distinct par seed
        self.norm_s = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in range(k)
        ])
        # LayerNorm de z 
        self.norm_z = nn.LayerNorm(emb_dim) 

        # Combinaison (s_i, z) → h_i — poids distincts par seed
        self.combine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * emb_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(k)
        ])

        # MLP final sur la concaténation h1...hk
        self.head = nn.Sequential(
            nn.Linear(k * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s, z):
        # s : (B, k, emb_dim)
        # z : (B, emb_dim)

        if s.dim() == 2:
            s = s.unsqueeze(1)   # (B, emb_dim) → (B, 1, emb_dim)

        hs = []
        z = self.norm_z(z)  
        for i in range(self.k):
            s_i = self.norm_s[i](s[:, i, :])      # (B, emb_dim)
            sz_i = torch.cat([s_i, z], dim=-1)    # (B, 2*emb_dim)
            hs.append(self.combine[i](sz_i))      # (B, hidden_dim)

        h_flat = torch.cat(hs, dim=-1)            # (B, k*hidden_dim)
        return self.head(h_flat)                  # (B, out_dim)


class ABMILPosPredictor(nn.Module):
    """
    Prédit la position (x, y) d'une vue z à partir des k seeds
    et de l'embedding DINO de cette vue.

    Poids et LayerNorm DISTINCTS par seed.

    s : (B, k, emb_dim)  — seeds
    z : (B, emb_dim)     — embedding DINO de la vue à localiser

    Logique ABMIL dans la couche intermédiaire
    """
    def __init__(self, emb_dim=768, k=1, hidden_dim=512, out_dim=2):
        super().__init__()
        self.k = k

        # LayerNorm distinct par seed
        self.norm_s = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in range(k)
        ])
        # LayerNorm de z 
        self.norm_z = nn.LayerNorm(emb_dim) 

        '''self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )'''

        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),   # ← 2*emb_dim au lieu de emb_dim
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Combinaison (s_i, z) → h_i — poids distincts par seed
        self.combine = nn.Sequential(
                nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.seed_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(k)
        ])

        self.z_transform = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # MLP final sur la concaténation h1...hk
        '''self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )'''

        self.head = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, s, z):
        # s : (B, k, emb_dim)
        # z : (B, emb_dim)

        if s.dim() == 2:
            s = s.unsqueeze(1)   # (B, emb_dim) → (B, 1, emb_dim)

        z_norm = self.z_transform(self.norm_z(z) )

        s_norm = torch.stack([
                                self.seed_transform[i](self.norm_s[i](s[:, i, :])) for i in range(self.k)
                            ], dim=1)   # (B, k, emb_dim)
        
        # Concaténer z à chaque seed pour le calcul des scores
        z_exp = z_norm.unsqueeze(1).expand(-1, self.k, -1) # (B, k, emb_dim)
        sz = torch.cat([s_norm, z_exp], dim=-1)            # (B, k, 2*emb_dim)

        # ABMIL conditionné sur z
        w = self.attn(sz)                                  # (B, k, 1)
        w = torch.softmax(w, dim=1)
        h = (w * s_norm).sum(dim=1)

        hz = torch.cat([h, z_norm], dim=-1)
        out = self.combine(hz)
        return self.head(out), w


        '''for i in range(self.k):
            s_i = self.norm_s[i](s[:, i, :])      # (B, emb_dim)
            sz_i = torch.cat([s_i, z], dim=-1)    # (B, 2*emb_dim)
            hs.append(self.combine[i](sz_i))      # (B, hidden_dim)
        hs = torch.stack(hs, dim=1)               # (B, k, hidden_dim)
        w = self.attn(hs)                         # (B, k, 1)
        w = torch.softmax(w, dim=1)
        h = (w * hs).sum(dim=1)              # (B, hidden_dim)
        return self.head(h)                  # (B, out_dim)'''

class TransformerPosPredictor(nn.Module):
    """
    Prédit la position (x, y) d'une vue z à partir des k seeds.
    Q = z (vue fovéale), K = V = seeds.
    Pas de résiduelle sur Q pour éviter que z influence directement la sortie.
    
    s : (B, k, emb_dim)
    z : (B, emb_dim)
    """

    def __init__(self, emb_dim=768, n_heads=8,
                hidden_dim=512, out_dim=2, dropout=0.1):
        super().__init__()

        self.z_transform = nn.Linear(emb_dim, hidden_dim)
        self.seed_transform = nn.Linear(emb_dim, hidden_dim) 

        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, s, z):
        if s.dim() == 2:
            s = s.unsqueeze(1)

        z = self.z_transform(z)               # (B, hidden_dim)

        s = self.seed_transform(s)            # (B, k, hidden_dim)

        q  = self.norm_q(z).unsqueeze(1)      # (B, 1, hidden_dim)
        kv = self.norm_kv(s)                  # (B, k, hidden_dim)

        h, attn_weights = self.cross_attn(q, kv, kv)       # (B, 1, hidden_dim)
        q = q + h
        q = q + self.ffn(self.norm_ffn(q))              # résiduelle FFN sur q ✓
        h = q.squeeze(1)                                # (B, hidden_dim)

        return self.head(self.norm_out(h)), attn_weights   # (B, out_dim), (B, 1, k)


class ABMILLabelPredictor(nn.Module):
    def __init__(self, emb_dim=768, k=1, hidden_dim=768,
                 n_classes=1000, label_emb_dim=32):
        super().__init__()
        self.k = k

        # Embedding du label
        self.label_embedding = nn.Embedding(n_classes, label_emb_dim)
        self.norm_label = nn.LayerNorm(label_emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, label_emb_dim))

        # LayerNorm et projection par seed
        self.norm_s = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(k)])
        self.seed_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(k)
        ])

        """self.label_transform = nn.Sequential(
                nn.Linear(label_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )"""

        # ABMIL : seed (hidden_dim) + label (label_emb_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim + label_emb_dim, 256),
            #nn.Linear(hidden_dim * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, s, labels=None):
        B = s.shape[0]
        if s.dim() == 2:
            s = s.unsqueeze(1)

        # ── Label encoding ────────────────────────────────────────────
        if labels is not None and self.training:
            mask = torch.rand(B, device=s.device) < 0.2
            l_emb = self.norm_label(self.label_embedding(labels))
            cls_exp = self.norm_label(self.cls_token.expand(B, -1))  # ← normalisé
            l_emb = torch.where(mask.unsqueeze(1), cls_exp, l_emb)
        elif labels is not None:
            l_emb = self.norm_label(self.label_embedding(labels))
        else:
            l_emb = self.norm_label(self.cls_token.expand(B, -1))    # ← normalisé
        # ── Projection des seeds ──────────────────────────────────────
        s_norm = torch.stack([
            #self.seed_transform[i](self.norm_s[i](s[:, i, :]))
            self.norm_s[i](s[:, i, :])
            for i in range(self.k)
        ], dim=1)                                      # (B, k, hidden_dim)

        # ── ABMIL conditionné sur le label ────────────────────────────
        l_exp = l_emb.unsqueeze(1).expand(-1, self.k, -1)  # (B, k, label_emb_dim)
        sl = torch.cat([s_norm.detach(), l_exp], dim=-1)        # (B, k, hidden_dim+label_emb_dim)

        w = self.attn(sl)                              # (B, k, 1)
        w = torch.softmax(w, dim=1)

        s_trans = torch.stack([
            self.seed_transform[i](s_norm[:, i, :])
            for i in range(self.k)
        ], dim=1)  

        h = (w * s_norm).sum(dim=1)                    # (B, hidden_dim)

        # ── Combinaison + prédiction ──────────────────────────────────

        return self.head(h), w                        # (B, n_classes), (B, k, 1)
    
class TransformerLabelHead(nn.Module):
    """
    Cross-attention : label (Q) × seeds (K, V)
    Le token label queye les seeds pour extraire
    l'information pertinente pour la classification.
    
    s      : (B, k, emb_dim)
    labels : (B,) — entiers
    """
    def __init__(self, emb_dim=768, n_heads=8, 
                 n_classes=1000, pretrained_embeddings=None, softmax=False, dropout=0.1, n_iter=3):
        super().__init__()

        # Token label — embedding appris
        if pretrained_embeddings is not None:
            n_classes, label_emb_dim = pretrained_embeddings.shape
            self.label_embedding = nn.Embedding(n_classes, label_emb_dim)
            self.label_embedding.weight.data.copy_(pretrained_embeddings)
            # Optionnel : geler les embeddings
            self.label_embedding.weight.requires_grad = False
        else:
            self.label_embedding = nn.Embedding(n_classes, emb_dim)
        #self.norm_label = nn.LayerNorm(emb_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Projection label_emb_dim → emb_dim si nécessaire
        # Ici on travaille directement en emb_dim pour la cohérence
        # avec les seeds

        # Norms Pre-LN
        self.norm_q  = nn.LayerNorm(emb_dim)   # sur le token label
        self.norm_kv = nn.LayerNorm(emb_dim)   # sur les seeds

        # Cross-attention : label (Q) × seeds (K, V)
        self.cross_attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )

        # FFN sur le token label après cross-attention
        self.norm_ffn = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

        # Tête de classification
        self.norm_out = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)
        self.n_iter = n_iter
        self.softmax = softmax

    def forward(self, s, labels=None):
        """
        s      : (B, k, emb_dim)
        labels : (B,) entiers — None à l'inférence
        """
        B = s.shape[0]

        # ── Token label ───────────────────────────────────────────────
        if labels is not None and self.training:
            mask   = torch.rand(B, device=s.device) < 0.2
            l_emb  = self.label_embedding(labels)  # (B, emb_dim)
            cls    = self.cls_token.expand(B, -1, -1).squeeze(1)
            l_emb  = torch.where(mask.unsqueeze(1), cls, l_emb)
        elif labels is not None:
            l_emb  = self.label_embedding(labels)  # (B, emb_dim)
        else:
            l_emb  = self.cls_token.expand(B, -1, -1).squeeze(1)

        q = l_emb.unsqueeze(1)             # (B, 1, emb_dim) — token unique
        # ── Pre-LN ───────────────────────────────────────────────────
        q_norm  = self.norm_q(q)           # (B, 1, emb_dim)
        kv_norm = self.norm_kv(s)          # (B, k, emb_dim)

        if labels is not None:
            # ── Cross-attention : label querye les seeds ───────────────────
            # Q = label token, K = V = seeds
            # Pas de self-attention → pas de leak possible
            h, attn_weights = self.cross_attn(q_norm, kv_norm, kv_norm)

            # Pas de résiduelle — sortie pure de la cross-attention
            out = self.ffn(self.norm_ffn(h))          # (B, 1, emb_dim)
            z   = self.norm_out(out.squeeze(1))       # (B, emb_dim)
            
        else: # Bootstrap sur les labels
            for i in range(self.n_iter):
                h, attn_weights = self.cross_attn(q_norm, kv_norm, kv_norm)
                out = self.ffn(self.norm_ffn(h)) 
                z   = self.norm_out(out.squeeze(1))  
                if i < self.n_iter - 1:
                    logits = self.head(z)
                    if self.softmax:
                        probs = torch.softmax(logits, dim=-1)      # (B, n_classes)
                        l_emb = probs @ self.label_embedding.weight
                    else:
                        pred_labels = logits.argmax(dim=-1)
                        l_emb = self.label_embedding(pred_labels)  # (B, label_emb_dim)
                    q = l_emb.unsqueeze(1) 
                    q_norm  = self.norm_q(q) 
        return self.head(z), attn_weights
        
class TransformerMixedHead(nn.Module):
    """
    Cross-attention : label (Q) × seeds (K, V)
    Le token label queye les seeds pour extraire
    l'information pertinente pour la classification.
    
    s      : (B, k, emb_dim)
    labels : (B,) — entiers
    """
    def __init__(self, emb_dim=768, n_heads=8, 
                 n_classes=1000, pretrained_embeddings=None,  dropout=0.1):
        super().__init__()

        # Token label — embedding appris
        if pretrained_embeddings is not None:
            n_classes, label_emb_dim = pretrained_embeddings.shape
            self.label_embedding = nn.Embedding(n_classes, label_emb_dim)
            self.label_embedding.weight.data.copy_(pretrained_embeddings)
            # Optionnel : geler les embeddings
            self.label_embedding.weight.requires_grad = False
        else:
            self.label_embedding = nn.Embedding(n_classes, emb_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Norms Pre-LN
        self.norm_q  = nn.LayerNorm(emb_dim)   # sur le token label
        self.norm_kv = nn.LayerNorm(emb_dim)   # sur les seeds

        # Cross-attention : label (Q) × seeds (K, V)
        self.cross_attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )

        # FFN sur le token label après cross-attention
        self.norm_ffn = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

        # Tête de classification
        self.norm_out = nn.LayerNorm(emb_dim)
        self.label_head = nn.Linear(emb_dim, n_classes)
        self.pos_head = nn.Linear(emb_dim, 2)

    def forward(self, s, z, labels=None):
        """
        s      : (B, k, emb_dim)
        labels : (B,) entiers — None à l'inférence
        """
        B = s.shape[0]

        # ── Token label ───────────────────────────────────────────────
        if labels is not None and self.training:
            mask   = torch.rand(B, device=s.device) < 0.2
            l_emb  = self.label_embedding(labels)  # (B, emb_dim)
            cls    = self.cls_token.expand(B, -1, -1).squeeze(1)
            l_emb  = torch.where(mask.unsqueeze(1), cls, l_emb)
        elif labels is not None:
            l_emb  = self.label_embedding(labels)  # (B, emb_dim)
        else:
            l_emb  = self.cls_token.expand(B, -1, -1).squeeze(1)

        # (B, 2, emb_dim) — token label + token z
        q = torch.stack([l_emb, z], dim=1)
        q_norm  = self.norm_q(q)                              # norm partagée ✓
        kv_norm = self.norm_kv(s)

        h, attn_weights = self.cross_attn(q_norm, kv_norm, kv_norm)  # (B, 2, emb_dim)
        out      = self.ffn(self.norm_ffn(h))                 # (B, 2, emb_dim)
        out_norm = self.norm_out(out)                         # (B, 2, emb_dim)

        return (self.label_head(out_norm[:, 0, :]),   # (B, n_classes)
                self.pos_head(out_norm[:, 1, :]),      # (B, 2)
                attn_weights)                          # (B, 2, k)
    

      
            
        
        

# class ABMILLabelPredictor(nn.Module):
#     """
#     Prédit le label ImageNet depuis les k seeds.
#     Le label (encodé) est injecté dans self.attn pour conditionner
#     la sélection des seeds — mais à l'inférence, le label n'est pas
#     disponible, donc on peut aussi utiliser un token CLS appris.
    
#     s : (B, k, emb_dim)
#     l : (B,) indices de labels entiers — optionnel à l'inférence
#     """
#     def __init__(self, emb_dim=768, k=1, hidden_dim=768,
#                  n_classes=1000, label_emb_dim=128, label_hidden_dim=768):
#         super().__init__()
#         self.k = k
#         self.label_emb_dim = label_emb_dim

#         # Embedding du label : 1000 classes → label_emb_dim
#         self.label_embedding = nn.Embedding(n_classes, label_emb_dim)
#         self.norm_label = nn.LayerNorm(label_emb_dim)

#         # Token CLS appris — utilisé à l'inférence quand le label est inconnu
#         self.cls_token = nn.Parameter(torch.randn(1, label_emb_dim))

#         # LayerNorm distinct par seed
#         self.norm_s = nn.ModuleList([
#             nn.LayerNorm(emb_dim) for _ in range(k)
#         ])

#         # Projection des seeds vers hidden_dim
#         self.seed_transform = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(emb_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#             )
#             for _ in range(k)
#         ])

#         # Projection du label vers hidden_dim
#         self.label_transform = nn.Sequential(
#             nn.Linear(label_emb_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#         # ABMIL conditionné sur le label
#         self.attn = nn.Sequential(
#             nn.Linear(2 * hidden_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 1),
#         )

#         # Combinaison seed agrégé + label
#         self.combine = nn.Sequential(
#             nn.Linear(2 * hidden_dim, 2 * hidden_dim),
#             nn.ReLU(),
#             nn.Linear(2 * hidden_dim, label_hidden_dim),
#             nn.ReLU(),
#         )

#         self.head = nn.Linear(label_hidden_dim, n_classes)

#     def forward(self, s, labels=None):
#         # s      : (B, k, emb_dim)
#         # labels : (B,) entiers — None à l'inférence
#         B = s.shape[0]

#         if s.dim() == 2:
#             s = s.unsqueeze(1)

#         # ── Encoding du label ─────────────────────────────────────────
#         if labels is not None:
#             l_emb = self.label_embedding(labels)       # (B, label_emb_dim)
#             l_emb = self.norm_label(l_emb)
#         else:
#             # À l'inférence : token CLS appris, partagé sur le batch
#             l_emb = self.cls_token.expand(B, -1)       # (B, label_emb_dim)

#         if labels is not None and self.training:
#             # Masquer aléatoirement le label pendant l'entraînement
#             mask = torch.rand(B, device=s.device) < 0.2   # 20% du temps
#             l_emb = self.label_embedding(labels)
#             cls_expanded = self.cls_token.expand(B, -1)
#             # Remplacer par cls_token là où le label est masqué
#             l_emb = torch.where(mask.unsqueeze(1), cls_expanded, l_emb)
#         elif labels is None:
#             l_emb = self.cls_token.expand(B, -1)

#         l_emb = self.norm_label(l_emb)

#         l_transformed = self.label_transform(l_emb)    # (B, hidden_dim)

#         # ── Projection des seeds ──────────────────────────────────────
#         s_norm = torch.stack([
#             self.seed_transform[i](self.norm_s[i](s[:, i, :]))
#             for i in range(self.k)
#         ], dim=1)                                       # (B, k, hidden_dim)

#         # ── ABMIL conditionné sur le label ────────────────────────────
#         l_exp = l_transformed.unsqueeze(1).expand(-1, self.k, -1)  # (B, k, hidden_dim)
#         sl = torch.cat([s_norm, l_exp], dim=-1)        # (B, k, 2*hidden_dim)

#         w = self.attn(sl)                              # (B, k, 1)
#         w = torch.softmax(w, dim=1)
#         h = (w * s_norm).sum(dim=1)                    # (B, hidden_dim)

#         # ── Combinaison + prédiction ──────────────────────────────────
#         hl = torch.cat([h, l_transformed], dim=-1)     # (B, 2*hidden_dim)
#         hl = self.combine(hl)                          # (B, hidden_dim)

#         return self.head(hl), w                        # (B, n_classes), (B, k, 1)
    
    
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
