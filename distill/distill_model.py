import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


class Down4x(nn.Module):
    """Learned 4x downsample: [B,3,H,W] -> [B,C,H/4,W/4] for any H,W divisible by 4."""

    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),  # H->H/2
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),  # H/2->H/4
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class BODistillation(nn.Module):
    """
    local_prior: [B,3,H,W]
    y:          [B,3,H/4,W/4]
    global S:   [B,1]
    -> outputs (g, l, gamma)
    """

    def __init__(self, bounds={"g": (0.0, 2.0), "l": (0.0, 2.0), "ga": (0.5, 2.0)},
                 width=64, m_ch=32):
        super().__init__()
        self.bounds = bounds

        self.m_down = Down4x(in_ch=3, out_ch=m_ch)

        # fuse y(3) + m_feat(m_ch) -> backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3 + m_ch, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),  # H/4 -> H/8
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),  # H/8 -> H/16
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(width * 4 + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(self, y, local_prior, s):
        """
        y:          [B,3,H/4,W/4]
        local_prior:[B,3,H,W]
        s:          [B,1]
        """
        m = self.m_down(local_prior)  # [B,m_ch,H/4,W/4]

        # (optional) assert shape matches exactly
        if (m.shape[-2:] != y.shape[-2:]):
            raise ValueError(f"m_down output {m.shape[-2:]} != y {y.shape[-2:]}. "
                             "Check H,W divisible by 4 and consistent preprocessing.")

        x = torch.cat([y, m], dim=1)  # [B,3+m_ch,H/4,W/4]
        f = self.backbone(x)
        print(f"=== f: {f.shape}")
        v = F.adaptive_avg_pool2d(f, 1).flatten(1)  # [B,width*4]
        print(f"=== v: {v.shape}")
        s = s.view(1, 1).expand(v.size(0), 1)
        print(f"=== s: {s.shape}")
        v = torch.cat([v, s], dim=1)
        print(f"=== v: {v.shape}")

        raw = torch.sigmoid(self.head(v))  # [B,3] in (0,1)

        gmin, gmax = self.bounds["g"]
        lmin, lmax = self.bounds["l"]
        gamin, gamax = self.bounds["ga"]
        g = gmin + raw[:, 0] * (gmax - gmin)
        l = lmin + raw[:, 1] * (lmax - lmin)
        ga = gamin + raw[:, 2] * (gamax - gamin)
        return g, l, ga
