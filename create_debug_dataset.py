"""
Script để tạo tập dataset debug với 100 ảnh train và 100 ảnh test
"""
import os
import shutil
from pathlib import Path

def create_debug_dataset(src_root, dest_root, num_samples=100):
    """
    Copy num_samples ảnh đầu tiên từ train/test vào thư mục debug
    
    Args:
        src_root: Thư mục gốc chứa dataset (ví dụ: datasets/ShanghaiTech/part_A)
        dest_root: Thư mục đích cho debug dataset
        num_samples: Số lượng ảnh muốn copy
    """
    src_root = Path(src_root)
    dest_root = Path(dest_root)
    
    for split in ['train_data', 'test_data']:
        print(f"\n=== Processing {split} ===")
        
        # Đường dẫn source và destination
        src_images = src_root / split / 'images'
        src_gts = src_root / split / 'ground-truth'
        
        dest_images = dest_root / split / 'images'
        dest_gts = dest_root / split / 'ground-truth'
        
        # Tạo thư mục đích
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_gts.mkdir(parents=True, exist_ok=True)
        
        # Lấy danh sách file ảnh
        image_files = sorted(src_images.glob('*.jpg'))
        
        # Giới hạn số lượng (nếu test_data có ít hơn num_samples thì lấy hết)
        actual_num = min(num_samples, len(image_files))
        image_files = image_files[:actual_num]
        
        print(f"Copying {actual_num} images from {split}...")
        
        # Copy từng ảnh và ground-truth tương ứng
        for img_path in image_files:
            # Copy ảnh
            shutil.copy2(img_path, dest_images / img_path.name)
            
            # Copy ground-truth (IMG_1.jpg -> GT_IMG_1.mat)
            gt_name = 'GT_' + img_path.stem + '.mat'
            gt_path = src_gts / gt_name
            
            if gt_path.exists():
                shutil.copy2(gt_path, dest_gts / gt_name)
            else:
                print(f"Warning: Ground-truth not found for {img_path.name}")
        
        print(f"✓ Copied {actual_num} images and ground-truth files")
    
    print(f"\n✓ Debug dataset created at: {dest_root}")
    print(f"  - Train: {min(num_samples, len(list((src_root / 'train_data' / 'images').glob('*.jpg'))))} samples")
    print(f"  - Test: {min(num_samples, len(list((src_root / 'test_data' / 'images').glob('*.jpg'))))} samples")

if __name__ == '__main__':
    # Cấu hình đường dẫn
    source_dataset = r'd:\Research\CrowCounting\SANET\datasets\ShanghaiTech\part_A'
    debug_dataset = r'd:\Research\CrowCounting\SANET\datasets\ShanghaiTech_debug\part_A'
    
    # Tạo debug dataset với 100 ảnh
    create_debug_dataset(source_dataset, debug_dataset, num_samples=100)
    
    print("\n" + "="*60)
    print("Để sử dụng debug dataset, hãy chỉnh tham số khi train:")
    print("  --train_images datasets/ShanghaiTech_debug/part_A/train_data/images")
    print("  --train_gts datasets/ShanghaiTech_debug/part_A/train_data/ground-truth")
    print("  --val_images datasets/ShanghaiTech_debug/part_A/test_data/images")
    print("  --val_gts datasets/ShanghaiTech_debug/part_A/test_data/ground-truth")
    print("="*60)
