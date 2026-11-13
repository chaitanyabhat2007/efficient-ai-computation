import numpy as np
H, W = 480, 640
S = 5000.0  # TUM depth scale

def flat_wall(z=2.0):
    return np.full((H, W), z * S)

def tilted_plane():
    u = np.arange(W)[None, :].repeat(H, 0)
    return (1.5 + 0.004 * (u - W / 2) + 1.0) * S

def sphere_on_wall():
    d = np.full((H, W), 3.0 * S)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    r2 = (u - W / 2) ** 2 + (v - H / 2) ** 2
    R = 120.0
    m = r2 < R ** 2
    d[m] = (2.0 - 0.6 * np.sqrt(1 - r2[m] / R ** 2)) * S
    return d

def random_valid(frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.integers(500, 5000, (H, W)).astype(np.float64)
    d[rng.random((H, W)) > frac] = 0
    return d

SCENES = {
    "flat_wall": flat_wall,
    "tilted_plane": tilted_plane,
    "sphere_on_wall": sphere_on_wall,
    "random_70valid": random_valid,
}
