import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ------------------------------
# Utils: Conv/Deconv blocks (Conv/Deconv + IN + ReLU)
# ------------------------------
class Conv2dIN(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=None, d=1, activate=True, use_norm=True, bias=False):
        super().__init__()
        if p is None:
            p = (k // 2) if isinstance(k, int) else 0
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True) if use_norm else None
        self.act = nn.ReLU(inplace=True) if activate else None

    def forward(self, x):
        x = self.conv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DeConv2dIN(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2, p=0, activate=True, use_norm=True, bias=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, output_padding=0, bias=bias)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True) if use_norm else None
        self.act = nn.ReLU(inplace=True) if activate else None

    def forward(self, x):
        x = self.deconv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ------------------------------
# Scale Aggregation (SA) Module: 4 nhánh 1/3/5/7; từ module thứ 2 trở đi có 1x1 reduce
# ------------------------------
class SAModule(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_reduction: bool):
        super().__init__()
        mid = out_c // 4  
        def make_branch(k: int, reduce: bool):
            if k == 1 or not reduce:
                return nn.Sequential(
                    Conv2dIN(in_c, mid, k, bias=False),
                )
            else:
                return nn.Sequential(
                    Conv2dIN(in_c, mid * 2, 1, bias=False),
                    Conv2dIN(mid * 2, mid, k, bias=False),
                )

        self.b1 = make_branch(1, use_reduction)
        self.b3 = make_branch(3, use_reduction)
        self.b5 = make_branch(5, use_reduction)
        self.b7 = make_branch(7, use_reduction)

        self.fuse = nn.Sequential(
            Conv2dIN(out_c, out_c, 1, bias=False),
        )

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        y = self.fuse(y)
        return y


# ------------------------------
# SANet (ECCV 2018)
# ------------------------------
class SANet(nn.Module):
    def __init__(self,
                 sa_channels=(64, 128, 256, 512), 
                 use_fuse_conv1x1=True,
                 ):
        super().__init__()

        c1, c2, c3, c4 = sa_channels

        # Feature Map Extraction (FME): 4 SA modules, pool sau 3 module đầu
        self.sa1 = SAModule(in_c=3, out_c=c1, use_reduction=False)   # module đầu: không reduce
        self.pool1 = nn.MaxPool2d(2, 2)

        self.sa2 = SAModule(in_c=c1, out_c=c2, use_reduction=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.sa3 = SAModule(in_c=c2, out_c=c3, use_reduction=True)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.sa4 = SAModule(in_c=c3, out_c=c4, use_reduction=True)   

        self.conv9  = Conv2dIN(c4, 512, 9, bias=False)
        self.conv7  = Conv2dIN(512, 512, 7, bias=False)
        self.conv5  = Conv2dIN(512, 256, 5, bias=False)
        self.conv3  = Conv2dIN(256, 128, 3, bias=False)
        
        # 3 Transposed conv layers (upsample 2x each)
        self.up1    = DeConv2dIN(128, 64, k=4, s=2, p=1)  # kernel=4, stride=2, padding=1 cho output size chính xác
        self.up2    = DeConv2dIN(64, 32, k=4, s=2, p=1)
        self.up3    = DeConv2dIN(32, 16, k=4, s=2, p=1)

        # Output 1x1 (không norm/act), sau đó ReLU để density ≥ 0
        self.out1x1 = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.relu_final = nn.ReLU(inplace=True)

        # Khởi tạo trọng số theo paper
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_size = x.shape[2:]  # (H, W)

        # Encoder (FME)
        x = self.sa1(x); x = self.pool1(x)  # /2
        x = self.sa2(x); x = self.pool2(x)  # /4
        x = self.sa3(x); x = self.pool3(x)  # /8
        x = self.sa4(x)  # stride tổng = 8

        x = self.conv9(x)
        x = self.conv7(x)
        x = self.conv5(x)
        x = self.conv3(x)
        
        x = self.up1(x)  # ×2 (stride=4)
        x = self.up2(x)  # ×4 (stride=2)
        x = self.up3(x)  # ×8 (stride=1, original size)
        
        x = self.out1x1(x)
        x = self.relu_final(x)

        if x.shape[2:] != inp_size:
            x = F.interpolate(x, size=inp_size, mode='bilinear', align_corners=False)
        return x


# ------------------------------
# SSIM (single-scale) cho loss
# ------------------------------
def _gaussian(window_size: int, sigma: float):
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
                          for x in range(window_size)], dtype=torch.float32)
    return gauss / gauss.sum()


def _create_window(window_size: int, sigma: float, channel: int, device, dtype):
    _1D = _gaussian(window_size, sigma).to(device=device, dtype=dtype).unsqueeze(1)
    _2D = _1D @ _1D.t()
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_single_scale(img1: torch.Tensor,
                      img2: torch.Tensor,
                      window_size: int = 11,
                      sigma: float = 1.5,
                      C1: float = 0.01**2,
                      C2: float = 0.03**2) -> torch.Tensor:
    """SSIM single-scale cho ảnh xám/density (B,1,H,W). Không backprop qua window."""
    assert img1.shape == img2.shape
    b, c, h, w = img1.shape
    device, dtype = img1.device, img1.dtype
    window = _create_window(window_size, sigma, c, device, dtype)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)


class SANetLoss(nn.Module):
    """
    L = MSE + alpha*(1 - SSIM) + beta*Count_Loss
    alpha=1e-3 (paper), beta=1e-3 (added for stability)
    """
    def __init__(self, alpha: float = 1e-3, beta: float = 1e-3, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B,1,H,W)
        
        # MSE loss (Euclidean loss)
        l2 = F.mse_loss(pred, target, reduction='mean')
        
        # SSIM loss
        ssim_val = ssim_single_scale(pred, target, window_size=self.window_size, sigma=self.sigma)  # (B,)
        l_ssim = 1.0 - ssim_val.mean()
        
        # Count loss (normalized by GT count to prevent explosion)
        pred_count = pred.view(pred.shape[0], -1).sum(dim=1)    # (B,)
        target_count = target.view(target.shape[0], -1).sum(dim=1)  # (B,)
        count_error = torch.abs(pred_count - target_count) / (target_count + 1.0)
        l_count = count_error.mean()
        
        # Total loss
        loss = l2 + self.alpha * l_ssim + self.beta * l_count
        
        return loss


# ------------------------------
# Sliding-window inference (patch-based, overlap 50%)
# ------------------------------
@torch.no_grad()
def sanet_sliding_window_inference(img: torch.Tensor,
                                   model: nn.Module,
                                   patch_ratio: float = 0.5,
                                   overlap: float = 0.5,
                                   gaussian_sigma_ratio: float = 0.125) -> torch.Tensor:

    single = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        single = True
    b, c, H, W = img.shape
    assert b == 1, "Hàm này demo theo paper chạy từng ảnh (B=1). Hãy lặp bên ngoài cho batch>1."

    device = img.device
    patch_h = max(32, int(H * patch_ratio))
    patch_w = max(32, int(W * patch_ratio))
    step_h = max(1, int(patch_h * (1.0 - overlap)))
    step_w = max(1, int(patch_w * (1.0 - overlap)))

    # Tạo trọng số Gaussian 2D cho blending (ưu tiên tâm patch)
    yy = torch.arange(patch_h, device=device).float()
    xx = torch.arange(patch_w, device=device).float()
    yy = (yy - (patch_h - 1) / 2.0) / (patch_h)
    xx = (xx - (patch_w - 1) / 2.0) / (patch_w)
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    sigma_y = gaussian_sigma_ratio
    sigma_x = gaussian_sigma_ratio
    weight = torch.exp(-(X**2) / (2 * sigma_x**2) - (Y**2) / (2 * sigma_y**2))
    weight = weight / (weight.max() + 1e-8)  # normalize to 1
    weight = weight.unsqueeze(0).unsqueeze(0)  # (1,1,Ph,Pw)

    out_acc = torch.zeros((1, 1, H, W), device=device)
    w_acc = torch.zeros((1, 1, H, W), device=device)

    for top in range(0, max(1, H - patch_h + 1), step_h):
        for left in range(0, max(1, W - patch_w + 1), step_w):
            patch = img[:, :, top:top + patch_h, left:left + patch_w]
            # Nếu chạm biên phải/dưới, dịch để đảm bảo phủ hết ảnh
            if patch.shape[2] < patch_h or patch.shape[3] < patch_w:
                top = H - patch_h
                left = W - patch_w
                patch = img[:, :, top:top + patch_h, left:left + patch_w]

            pred = model(patch)  # (1,1,Ph,Pw)
            out_acc[:, :, top:top + patch_h, left:left + patch_w] += pred * weight
            w_acc[:, :, top:top + patch_h, left:left + patch_w] += weight

    out = out_acc / (w_acc + 1e-8)
    if single:
        out = out.squeeze(0)
    return out


# ------------------------------
# Ví dụ sử dụng
# ------------------------------
if __name__ == "__main__":
    # Dummy test
    net = SANet().cuda() if torch.cuda.is_available() else SANet()
    x = torch.randn(1, 3, 512, 512, device=next(net.parameters()).device)
    y = net(x)
    print("Output:", y.shape)  # (1,1,512,512)

    # Loss demo
    criterion = SANetLoss(alpha=1e-3)
    target = torch.abs(torch.randn_like(y))
    loss = criterion(y, target)
    print("Loss:", float(loss))

    # Sliding-window inference demo (B=1):
    y_sw = sanet_sliding_window_inference(x, net, patch_ratio=0.5, overlap=0.5)
    print("SW Output:", y_sw.shape)
