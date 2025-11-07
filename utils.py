


import os, glob, math, random, argparse
import numpy as np
import cv2
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm




# ---------- Utils ----------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_image_mat_pairs(img_dir, gt_dir, img_exts=(".jpg", ".jpeg", ".png")):
    img_paths = []
    for ext in img_exts:
        img_paths += sorted(glob.glob(os.path.join(img_dir, f"*{ext}")))
    pairs = []
    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        # ShanghaiTech: IMG_1.jpg <-> GT_IMG_1.mat
        gt = os.path.join(gt_dir, f"GT_{base}.mat")
        if not os.path.isfile(gt):
            raise FileNotFoundError(f"Không tìm thấy GT cho {ip}: {gt}")
        pairs.append((ip, gt))
    return pairs

def load_points_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    if 'image_info' in mat:
        points = mat['image_info'][0, 0][0, 0][0]
    elif 'annPoints' in mat:
        points = mat['annPoints']
    else:
        raise KeyError(f"Không tìm thấy key 'image_info' hay 'annPoints' trong {mat_path}")
    pts = np.array(points, dtype=np.float32)
    # đảm bảo dạng [(x,y), ...]
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    return pts.tolist()

def make_density_map_fixed_sigma(img_h, img_w, points_xy, sigma=8):
    dm = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points_xy) == 0:
        return dm
    for x, y in points_xy:
        xi = min(img_w - 1, max(0, int(round(x))))
        yi = min(img_h - 1, max(0, int(round(y))))
        dm[yi, xi] += 1.0
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0: ksize += 1
    dm = cv2.GaussianBlur(dm, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    return dm

def gaussian_kernel(win_size=11, sigma=1.5, channels=1, device='cpu'):
    coords = torch.arange(win_size, dtype=torch.float32, device=device) - win_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)  # 1xW
    kernel_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)    # Cx1xHxW
    return kernel_2d

def ssim_map(pred, gt, win_size=11, sigma=1.5):
    # pred, gt: [B,1,H,W]
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = pred.shape
    device = pred.device
    kernel = gaussian_kernel(win_size, sigma, C, device)
    mu_x = F.conv2d(pred, kernel, padding=win_size//2, groups=C)
    mu_y = F.conv2d(gt,   kernel, padding=win_size//2, groups=C)
    mu_x2, mu_y2 = mu_x*mu_x, mu_y*mu_y
    mu_xy = mu_x*mu_y
    sigma_x  = F.conv2d(pred*pred, kernel, padding=win_size//2, groups=C) - mu_x2
    sigma_y  = F.conv2d(gt*gt,     kernel, padding=win_size//2, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred*gt,   kernel, padding=win_size//2, groups=C) - mu_xy
    ssim_n = (2*mu_xy + C1) * (2*sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim   = ssim_n / (ssim_d + 1e-12)
    return ssim

class SANetLoss(nn.Module):
    def __init__(self, alpha_c=1e-3, win_size=11, sigma=1.5):
        super().__init__()
        self.alpha_c = alpha_c
        self.win_size = win_size
        self.sigma = sigma
    def forward(self, pred, gt):
        le = F.mse_loss(pred, gt, reduction='mean')
        ssim = ssim_map(pred, gt, win_size=self.win_size, sigma=self.sigma)
        lc = 1.0 - ssim.mean()
        return le + self.alpha_c * lc, {'LE': le.item(), 'LC': lc.item()}


# ---------- Dataset ----------
class CrowdPatchDataset(Dataset):
    def __init__(self, pairs, sigma=8, train=True):
        self.pairs = pairs
        self.sigma = sigma
        self.train = train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        pts = load_points_from_mat(gt_path)
        dm = make_density_map_fixed_sigma(H, W, pts, sigma=self.sigma)

        # patch 1/4 size = (H/2, W/2)
        ph, pw = max(2, H//2), max(2, W//2)
        if self.train:
            y0 = random.randint(0, max(0, H - ph))
            x0 = random.randint(0, max(0, W - pw))
            img = img[y0:y0+ph, x0:x0+pw]
            dm  = dm [y0:y0+ph, x0:x0+pw]
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                dm  = dm [:, ::-1].copy()
        else:
            # với val loader ta cũng lấy 1 patch ngẫu nhiên (nhẹ) để logging nhanh
            y0 = (H - ph)//2 if H>=ph else 0
            x0 = (W - pw)//2 if W>=pw else 0
            img = img[y0:y0+ph, x0:x0+pw]
            dm  = dm [y0:y0+ph, x0:x0+pw]

        img = (img / 255.0).astype(np.float32)
        img = torch.from_numpy(img.transpose(2,0,1))  # CxHxW
        dm  = torch.from_numpy(dm).unsqueeze(0)       # 1xHxW
        return img, dm


# ---------- Tiled inference theo paper ----------
def tile_infer_patch_quarter(model, img_np):
    model.eval()
    H, W, _ = img_np.shape
    ph, pw = max(2, H//2), max(2, W//2)       # 1/4 size
    sy, sx = max(1, ph//2), max(1, pw//2)     # 50% overlap

    out_full = np.zeros((H, W), dtype=np.float32)
    dist_full = np.full((H, W), np.inf, dtype=np.float32)

    with torch.no_grad():
        for y in range(0, max(1, H - ph + 1), sy):
            for x in range(0, max(1, W - pw + 1), sx):
                patch = img_np[y:y+ph, x:x+pw]
                tin = torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0).float().to(next(model.parameters()).device)
                pred = model(tin).squeeze(0).squeeze(0).cpu().numpy()  # ph x pw

                yy, xx = np.mgrid[0:ph, 0:pw]
                cy, cx = ph/2.0, pw/2.0
                d = np.sqrt((yy-cy)**2 + (xx-cx)**2)

                sub_d = dist_full[y:y+ph, x:x+pw]
                sub_o = out_full [y:y+ph, x:x+pw]
                mask = d < sub_d
                sub_o[mask] = pred[mask]
                sub_d[mask] = d[mask]
                out_full[y:y+ph, x:x+pw] = sub_o
                dist_full[y:y+ph, x:x+pw] = sub_d
    return out_full
