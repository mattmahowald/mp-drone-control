import torch, math, random

def _rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])

def geom_aug(vec: torch.Tensor,
             max_shift=.05, max_scale=.05, max_deg=15):
    """vec in R^63  (already wrist-centred & unit-scaled)."""
    pts = vec.view(-1, 3)               # (21,3)
    # scale
    scale = 1 + random.uniform(-max_scale, max_scale)
    pts[:, :2] *= scale
    # rotate
    th = math.radians(random.uniform(-max_deg, max_deg))
    pts[:, :2] = pts[:, :2] @ _rot2d(th).to(pts)
    # translate
    dx, dy = (random.uniform(-max_shift, max_shift) for _ in range(2))
    pts[:, 0] += dx; pts[:, 1] += dy
    return pts.flatten()

def jitter(vec: torch.Tensor, sigma=.005):
    return vec + sigma * torch.randn_like(vec)



