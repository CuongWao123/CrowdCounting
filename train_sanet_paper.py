"""
Training script for SANet - Theo đúng paper ECCV 2018
Paper: Scale Aggregation Network for Accurate and Efficient Crowd Counting

Training Details từ Paper:
- Patch size: 1/4 original image (H/2, W/2)
- Data augmentation: Random crop + horizontal flip (p=0.5)
- Density map: Fixed Gaussian σ = 8 (not geometry-adaptive)
- Optimizer: Adam, lr = 1e-5, weight_decay = 0
- Weight init: Gaussian(mean=0, std=0.01)
- Loss: L_E (MSE) + α_c * L_C (SSIM), α_c = 1e-3, win_size=11, σ_ssim=1.5
- Evaluation: Tiled inference với 50% overlap
"""

import os, math, argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import SANet
from utils import (
    set_seed, list_image_mat_pairs, load_points_from_mat,
    tile_infer_patch_quarter, CrowdPatchDataset, SANetLoss
)

def evaluate_full_images(model, pairs, device='cuda'):
    mae, se = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for img_path, gt_path in tqdm(pairs, desc="Evaluating", leave=False):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Cannot read {img_path}")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred_dm = tile_infer_patch_quarter(model, img)
            C_pred = float(pred_dm.sum())
            C_gt = float(len(load_points_from_mat(gt_path)))
            mae += abs(C_pred - C_gt)
            se += (C_pred - C_gt) ** 2
    N = max(1, len(pairs))
    return {'MAE': mae / N, 'MSE': math.sqrt(se / N)}

def train(args):
    # Seed
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Data
    print("\n=== Loading Dataset ===")
    train_pairs = list_image_mat_pairs(args.train_images, args.train_gts)
    val_pairs = list_image_mat_pairs(args.val_images, args.val_gts)
    print(f"Train: {len(train_pairs)} images")
    print(f"Val: {len(val_pairs)} images")
    
    train_ds = CrowdPatchDataset(train_pairs, sigma=args.sigma, train=True)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == 'cuda'),
        drop_last=True
    )
    
    # Model
    print("\n=== Initializing Model ===")
    model = SANet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss & Optimizer (theo paper)
    criterion = SANetLoss(alpha_c=1e-3, win_size=11, sigma=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # AMP (optional)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == 'cuda')
    
    # Resume
    start_epoch = 1
    best_mae = float('inf')
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get('model', ckpt))
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'best_mae' in ckpt:
            best_mae = ckpt['best_mae']
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
    
    # Logging
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'train_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch\tLoss\tLE\tLC\tValMAE\tValMSE\n")
    
    # TensorBoard (optional)
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'runs'))
            print(f"TensorBoard log dir: {os.path.join(args.out_dir, 'runs')}")
        except:
            print("[WARN] TensorBoard not available")
    
    # Training Loop
    print(f"\n=== Training from epoch {start_epoch} to {args.epochs} ===\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        epoch_loss = 0.0
        epoch_le = 0.0
        epoch_lc = 0.0
        n_batches = 0
        
        for img, dm in pbar:
            img = img.to(device, non_blocking=True)
            dm = dm.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward
            if args.amp and device == 'cuda':
                with torch.cuda.amp.autocast():
                    pred = model(img)
                    loss, parts = criterion(pred, dm)
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(img)
                loss, parts = criterion(pred, dm)
                loss.backward()
                
                # Gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            # Stats
            epoch_loss += loss.item()
            epoch_le += parts['LE']
            epoch_lc += parts['LC']
            n_batches += 1
            
            # Debug: track prediction stats
            with torch.no_grad():
                pred_count = pred.sum().item()
                gt_count = dm.sum().item()
            
            # Progress bar
            pbar.set_postfix({
                'Loss': f"{epoch_loss/n_batches:.4f}",
                'LE': f"{epoch_le/n_batches:.4f}",
                'LC': f"{epoch_lc/n_batches:.4f}",
                'PredC': f"{pred_count:.1f}",
                'GTC': f"{gt_count:.1f}"
            })
        
        avg_loss = epoch_loss / n_batches
        avg_le = epoch_le / n_batches
        avg_lc = epoch_lc / n_batches
        
        # Evaluation
        val_mae = val_mse = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n[Epoch {epoch}] Evaluating...")
            metrics = evaluate_full_images(model, val_pairs, device=device)
            val_mae = metrics['MAE']
            val_mse = metrics['MSE']
            print(f"[Epoch {epoch}] VAL - MAE: {val_mae:.2f}, MSE: {val_mse:.2f}")
            
            # Save best
            if val_mae < best_mae:
                best_mae = val_mae
                best_path = os.path.join(args.out_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_mae': best_mae,
                    'val_mse': val_mse
                }, best_path)
                print(f"✓ Saved best model (MAE={best_mae:.2f})")
        
        # Logging
        with open(log_file, 'a') as f:
            f.write(f"{epoch}\t{avg_loss:.6f}\t{avg_le:.6f}\t{avg_lc:.6f}\t")
            f.write(f"{val_mae if val_mae else 'N/A'}\t{val_mse if val_mse else 'N/A'}\n")
        
        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar('Loss/train', avg_loss, epoch)
            tb_writer.add_scalar('Loss/LE', avg_le, epoch)
            tb_writer.add_scalar('Loss/LC', avg_lc, epoch)
            if val_mae:
                tb_writer.add_scalar('MAE/val', val_mae, epoch)
                tb_writer.add_scalar('MSE/val', val_mse, epoch)
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae
            }, ckpt_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")
    
    if tb_writer:
        tb_writer.close()
    
    print(f"\n=== Training Complete ===")
    print(f"Best MAE: {best_mae:.2f}")
    print(f"Models saved in: {args.out_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train SANet (ECCV 2018)')
    
    # Dataset
    parser.add_argument('--train-images', type=str, required=True, help='Path to training images')
    parser.add_argument('--train-gts', type=str, required=True, help='Path to training ground truth')
    parser.add_argument('--val-images', type=str, required=True, help='Path to validation images')
    parser.add_argument('--val-gts', type=str, required=True, help='Path to validation ground truth')
    
    # Training params (theo paper)
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (1 for variable patch sizes)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (paper: 1e-5)')
    parser.add_argument('--sigma', type=float, default=8.0, help='Gaussian sigma for density map (paper: 8)')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    
    # System
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    
    # Checkpointing
    parser.add_argument('--out-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N epochs')
    
    # Logging
    parser.add_argument('--tensorboard', action='store_true', help='Use TensorBoard logging')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
