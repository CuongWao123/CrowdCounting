import cv2
import numpy as np
from utils import load_points_from_mat, make_density_map_fixed_sigma
import matplotlib.pyplot as plt

# Test với 1 ảnh
img_path = "datasets/ShanghaiTech_debug/part_A/train_data/images/IMG_1.jpg"
gt_path = "datasets/ShanghaiTech_debug/part_A/train_data/ground-truth/GT_IMG_1.mat"

img = cv2.imread(img_path)
H, W = img.shape[:2]

pts = load_points_from_mat(gt_path)
print(f"Số người (GT): {len(pts)}")

# Tạo density map với sigma = 8
dm = make_density_map_fixed_sigma(H, W, pts, sigma=8)
print(f"Sum của density map: {dm.sum():.2f}")
print(f"Max value: {dm.max():.4f}")
print(f"Min value: {dm.min():.4f}")

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f'Original Image\nGT Count: {len(pts)}')
plt.subplot(132)
plt.imshow(dm, cmap='jet')
plt.colorbar()
plt.title(f'Density Map\nSum: {dm.sum():.2f}')
plt.subplot(133)
plt.hist(dm.flatten(), bins=100)
plt.title('Density Value Distribution')
plt.xlabel('Density Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('density_map_check.png', dpi=150)
print("Saved to density_map_check.png")