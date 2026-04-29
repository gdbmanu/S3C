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