#!/usr/bin/env python3
"""
scripts/train_runner.py
Skrip pelatihan dan evaluasi model estimasi usia wajah (MobileNetV3-Small) pada UTKFace.
Mendukung pelatihan bertahap (B1 -> M1), regularisasi adaptif, dan metrik age-gating.
"""

import os
import sys
import re
import io
import json
import math
import time
import shutil
import hashlib
import random
import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / 'manifest'
DATA_DIR = ROOT / 'data' / 'utkface'
RUNS_DIR = ROOT / 'runs'
RESULTS_DIR = ROOT / 'results'
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGE_RANGE = (0, 116)
AGE_BINS = [(0, 12), (13, 17), (18, 24), (25, 39), (40, 59), (60, 116)]
AGE_BIN_LABELS = [f'{lo}-{hi}' if hi < AGE_RANGE[1] else f'{lo}+' for lo, hi in AGE_BINS]
GENDER_NAMES = {0: 'laki-laki', 1: 'perempuan'}
RACE_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
EYE_COLUMNS = ['eye_right_x', 'eye_right_y', 'eye_left_x', 'eye_left_y']
GLASSES_COLORS = [(20, 20, 20), (45, 35, 30), (70, 45, 25), (110, 110, 115), (175, 150, 90), (30, 30, 60)]

def now_utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

def age_bin(age):
    for (lo, hi), label in zip(AGE_BINS, AGE_BIN_LABELS):
        if lo <= age <= hi:
            return label
    raise ValueError(age)

def draw_glasses(image, eyes, rng=random):
    image = image.convert('RGBA')
    right_x, right_y, left_x, left_y = eyes
    eye_dist = math.hypot(left_x - right_x, left_y - right_y)
    if eye_dist < 5:
        return image.convert('RGB')
    mid_x, mid_y = (right_x + left_x) / 2, (right_y + left_y) / 2
    angle = math.degrees(math.atan2(left_y - right_y, left_x - right_x))
    lens_w = eye_dist * rng.uniform(0.72, 0.95)
    lens_h = lens_w * rng.uniform(0.55, 0.85)
    thickness = max(1, round(eye_dist * rng.uniform(0.04, 0.10)))
    color = rng.choice(GLASSES_COLORS) + (rng.randint(220, 255),)
    tint = (rng.randint(180, 255),) * 3 + (rng.randint(0, 35),)
    lens_y = mid_y + eye_dist * rng.uniform(-0.05, 0.08)
    layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rounded = rng.random() < 0.6
    for center_x in (mid_x - eye_dist / 2, mid_x + eye_dist / 2):
        box = [center_x - lens_w / 2, lens_y - lens_h / 2, center_x + lens_w / 2, lens_y + lens_h / 2]
        if rounded:
            draw.rounded_rectangle(box, radius=lens_h * rng.uniform(0.2, 0.45), fill=tint, outline=color, width=thickness)
        else:
            draw.ellipse(box, fill=tint, outline=color, width=thickness)
    bridge_y = lens_y - lens_h * rng.uniform(0.0, 0.3)
    draw.line([(mid_x - eye_dist / 2 + lens_w / 2, bridge_y), (mid_x + eye_dist / 2 - lens_w / 2, bridge_y)],
              fill=color, width=thickness)
    temple_y = lens_y - lens_h * 0.3
    for side in (-1, 1):
        outer_x = mid_x + side * (eye_dist / 2 + lens_w / 2)
        draw.line([(outer_x, temple_y), (outer_x + side * eye_dist * 0.45, temple_y + eye_dist * 0.05)],
                  fill=color, width=thickness)
    layer = layer.rotate(-angle, resample=Image.BILINEAR, center=(mid_x, mid_y))
    return Image.alpha_composite(image, layer).convert('RGB')

class FaceAgeDataset(Dataset):
    def __init__(self, df, transform, glasses_p=0.0, glasses_seed=None):
        self.paths = [DATA_DIR / p for p in df.path]
        self.ages = df.age.to_numpy(dtype=np.float32)
        self.transform = transform
        self.glasses_p = glasses_p
        self.glasses_seed = glasses_seed
        has_eyes = all(col in df.columns for col in EYE_COLUMNS)
        self.eyes = df[EYE_COLUMNS].to_numpy(dtype=np.float32) if has_eyes and glasses_p > 0 else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            image = image.convert('RGB')
        if self.glasses_p > 0 and self.eyes is not None and not np.isnan(self.eyes[i]).any():
            rng = random.Random(self.glasses_seed + i) if self.glasses_seed is not None else random
            if rng.random() < self.glasses_p:
                image = draw_glasses(image, self.eyes[i], rng)
        return self.transform(image), self.ages[i]

def seed_worker(worker_id, base_seed):
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)

def make_loader(df, transform, shuffle, seed, batch_size, num_workers, sampler=None,
                glasses_p=0.0, glasses_seed=None, device_type='cpu'):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(FaceAgeDataset(df, transform, glasses_p, glasses_seed),
                      batch_size=batch_size,
                      shuffle=shuffle and sampler is None,
                      sampler=sampler,
                      num_workers=num_workers,
                      pin_memory=(device_type == 'cuda'),
                      generator=generator,
                      persistent_workers=(num_workers > 0),
                      worker_init_fn=functools.partial(seed_worker, base_seed=seed) if num_workers else None)

def build_model(median_age, dropout=0.3, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights, progress=False)
    age_head = nn.Linear(model.classifier[3].in_features, 1)
    nn.init.normal_(age_head.weight, 0.0, 0.01)
    nn.init.constant_(age_head.bias, median_age)
    model.classifier[3] = age_head
    model.classifier[2].p = dropout
    return model

def param_groups_with_decay(params, lr, weight_decay):
    return [{'params': [p for p in params if p.ndim > 1], 'lr': lr, 'weight_decay': weight_decay},
            {'params': [p for p in params if p.ndim <= 1], 'lr': lr, 'weight_decay': 0.0}]

def set_trainable_params(model, mode, lr_head, lr_backbone, weight_decay, unfreeze_last):
    for param in model.parameters():
        param.requires_grad = False
    head_params = list(model.classifier.parameters())
    for param in head_params:
        param.requires_grad = True
    param_groups = param_groups_with_decay(head_params, lr_head, weight_decay)
    if mode == 'finetune':
        backbone_params = [p for module in model.features[-unfreeze_last:] for p in module.parameters()]
        for param in backbone_params:
            param.requires_grad = True
        param_groups += param_groups_with_decay(backbone_params, lr_backbone, weight_decay)
    return param_groups

def set_train_mode(model, mode, unfreeze_last, update_bn_stats):
    model.eval()
    model.classifier.train()
    if mode == 'finetune' and update_bn_stats:
        for module in model.features[-unfreeze_last:]:
            module.train()

def lr_multiplier(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

@torch.inference_mode()
def predict_ages(model, df, transform, batch_size, num_workers, device, tta_flip=True):
    model.eval()
    loader = make_loader(df, transform, False, 0, batch_size=batch_size, num_workers=num_workers, device_type=device.type)
    preds = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        out = model(images).squeeze(1)
        if tta_flip:
            out = (out + model(torch.flip(images, dims=[3])).squeeze(1)) / 2.0
        preds.append(out.float().cpu())
    return torch.cat(preds).numpy().astype(np.float64)

def calculate_gating_metrics(y_true, y_pred, l_targets=(18, 21), fpr_target=0.05):
    from sklearn.metrics import roc_auc_score
    results = {}
    for L in l_targets:
        is_adult_true = (y_true >= L).astype(int)
        try:
            auc = float(roc_auc_score(is_adult_true, y_pred))
        except Exception:
            auc = 0.0
        # Cari T rekomendasi: T terkecil di mana FPR <= fpr_target
        # FPR = proportion of true minors who get predicted >= T
        best_t, best_fnr, best_fpr = L, 1.0, 1.0
        for T_cand in range(L, L + 16):
            pred_adult = (y_pred >= T_cand).astype(int)
            minor_mask = (y_true < L)
            adult_mask = (y_true >= L)
            fpr = float(np.mean(pred_adult[minor_mask])) if np.sum(minor_mask) > 0 else 0.0
            fnr = float(np.mean(1 - pred_adult[adult_mask])) if np.sum(adult_mask) > 0 else 0.0
            if fpr <= fpr_target:
                best_t = T_cand
                best_fpr = fpr
                best_fnr = fnr
                break
        results[f'L{L}_T_rec'] = best_t
        results[f'L{L}_FPR'] = round(best_fpr * 100, 2)
        results[f'L{L}_FNR'] = round(best_fnr * 100, 2)
        results[f'L{L}_AUC'] = round(auc, 4)
    return results

def train_one_run(run_name, method, mode, model, train_loader, val_df, eval_transform,
                  median_age, cfg, device, init_ckpt=None):
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if init_ckpt and os.path.exists(init_ckpt):
        ckpt_data = torch.load(init_ckpt, map_location='cpu')
        model.load_state_dict(ckpt_data['model'])
        print(f'[{run_name}] Checkpoint dimuat dari {init_ckpt}')

    param_groups = set_trainable_params(model, mode, cfg['lr_head'], cfg['lr_backbone'],
                                        cfg['weight_decay'], cfg['unfreeze_last'])
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg['weight_decay'])
    loss_fn = nn.SmoothL1Loss(beta=cfg['smooth_l1_beta'])

    total_steps = cfg['max_epochs'] * len(train_loader)
    warmup_steps = max(1, cfg['warmup_epochs'] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, warmup_steps, total_steps))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[{run_name}] Parameter yang dilatih: {n_trainable:,}')

    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []
    t_start = time.time()

    val_targets = val_df.age.to_numpy(dtype=np.float64)

    for epoch in range(1, cfg['max_epochs'] + 1):
        ep_t0 = time.time()
        set_train_mode(model, mode, cfg['unfreeze_last'], cfg['update_bn_stats'])
        loss_sum = abs_err_sum = 0.0
        n_seen = 0

        for images, ages in train_loader:
            images, ages = images.to(device, non_blocking=True), ages.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images).squeeze(1)
            loss = loss_fn(preds.float(), ages)
            loss.backward()
            optimizer.step()
            scheduler.step()

            n_batch = len(ages)
            loss_sum += loss.item() * n_batch
            abs_err_sum += (preds.detach().float() - ages).abs().sum().item()
            n_seen += n_batch

        train_loss = loss_sum / n_seen
        train_mae = abs_err_sum / n_seen

        val_preds = predict_ages(model, val_df, eval_transform, cfg['batch_size'],
                                 cfg['num_workers'], device, tta_flip=cfg.get('tta_flip', True))
        val_mae = float(np.mean(np.abs(val_preds - val_targets)))
        overfit_gap = val_mae - train_mae
        ep_sec = round(time.time() - ep_t0, 1)

        saved_msg = ''
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            saved_msg = '  (checkpoint disimpan *)'
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_mae': val_mae,
                        'train_mae': train_mae, 'method': method, 'cfg': cfg, 'median_age': median_age},
                       run_dir / 'best.pt')
        else:
            patience_counter += 1

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_mae': train_mae,
                        'val_mae': val_mae, 'overfit_gap': overfit_gap, 'seconds': ep_sec})
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)

        print(f'[{run_name}] Epoch {epoch:2d}/{cfg["max_epochs"]} | loss {train_loss:.3f} | '
              f'MAE latih {train_mae:.2f} | MAE val {val_mae:.2f} | gap {overfit_gap:+.2f} | {ep_sec}s{saved_msg}')

        if patience_counter >= cfg['patience']:
            print(f'[{run_name}] Early stopping dipicu pada epoch {epoch}')
            break

    total_min = round((time.time() - t_start) / 60, 2)
    done_info = {'best_val_mae': best_val_mae, 'best_epoch': best_epoch,
                 'epochs_run': len(history), 'train_minutes': total_min, 'finished_utc': now_utc()}
    with open(run_dir / 'done.json', 'w') as f:
        json.dump(done_info, f, indent=2)

    return run_dir, best_val_mae

def main():
    parser = argparse.ArgumentParser(description='UTKFace Age-Gating Model Trainer')
    parser.add_argument('--run-tag', type=str, default='r6_exp1')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr-head', type=float, default=3e-4)
    parser.add_argument('--lr-backbone', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--smooth-l1-beta', type=float, default=1.0)
    parser.add_argument('--aug-glasses-p', type=float, default=0.3)
    parser.add_argument('--aug-rotate', type=int, default=10)
    parser.add_argument('--aug-translate', type=float, default=0.05)
    parser.add_argument('--unfreeze-last', type=int, default=13)
    parser.add_argument('--balanced-sampling', action='store_true')
    parser.add_argument('--quick-eval-samples', type=int, default=0, help='Gunakan subset untuk iterasi cepat jika > 0')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'== Menjalankan Pelatihan [{args.run_tag}] di {device} ==')

    # Manifest
    manifest_path = MANIFEST_DIR / 'manifest.csv'
    if not manifest_path.exists():
        raise RuntimeError('Manifest belum dibuat! Jalankan scripts/prepare_manifest.py terlebih dahulu.')

    manifest_df = pd.read_csv(manifest_path)
    with open(manifest_path, 'rb') as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()

    # Load eyes landmarks
    eyes_path = MANIFEST_DIR / f'eyes_{manifest_sha[:8]}.csv'
    if eyes_path.exists():
        eyes_df = pd.read_csv(eyes_path)
        manifest_df = manifest_df.merge(eyes_df, on='path', how='left')

    train_df = manifest_df[manifest_df.split == 'train'].reset_index(drop=True)
    val_df = manifest_df[manifest_df.split == 'val'].reset_index(drop=True)

    if args.quick_eval_samples > 0:
        train_df = train_df.sample(n=min(args.quick_eval_samples, len(train_df)), random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(n=min(args.quick_eval_samples // 4, len(val_df)), random_state=args.seed).reset_index(drop=True)
        print(f'Quick mode: train={len(train_df)}, val={len(val_df)}')

    median_age = float(np.median(train_df.age))
    print(f'Train samples: {len(train_df)} | Val samples: {len(val_df)} | Median age: {median_age}')

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = T.Normalize(mean, std)
    eval_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomAffine(degrees=args.aug_rotate, translate=(args.aug_translate, args.aug_translate),
                       scale=(0.9, 1.1), fill=tuple(int(255 * v) for v in mean)),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        normalize
    ])

    sampler = None
    if args.balanced_sampling:
        bin_counts = train_df.age_bin.value_counts()
        weights = 1.0 / train_df.age_bin.map(bin_counts).to_numpy()
        sampler_gen = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                        num_samples=len(weights), replacement=True, generator=sampler_gen)

    train_loader = make_loader(train_df, train_transform, True, args.seed,
                               batch_size=args.batch_size, num_workers=args.num_workers,
                               sampler=sampler, glasses_p=args.aug_glasses_p, device_type=device.type)

    cfg = {
        'run_tag': args.run_tag,
        'seed': args.seed,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'warmup_epochs': 2,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'lr_head': args.lr_head,
        'lr_backbone': args.lr_backbone,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'smooth_l1_beta': args.smooth_l1_beta,
        'unfreeze_last': args.unfreeze_last,
        'update_bn_stats': True,
        'tta_flip': True
    }

    # Tahap 1: Latih Head (B1)
    b1_run_name = f'B1_s{args.seed}_{args.run_tag}'
    b1_model = build_model(median_age, dropout=args.dropout)
    b1_cfg = dict(cfg, max_epochs=min(15, args.epochs), lr_head=args.lr_head, unfreeze_last=0, update_bn_stats=False)
    b1_dir, b1_val_mae = train_one_run(b1_run_name, 'B1', 'head', b1_model, train_loader,
                                       val_df, eval_transform, median_age, b1_cfg, device)

    # Tahap 2: Fine-Tuning Seluruh Backbone (M1)
    m1_run_name = f'M1_s{args.seed}_{args.run_tag}'
    m1_model = build_model(median_age, dropout=args.dropout)
    m1_dir, m1_val_mae = train_one_run(m1_run_name, 'M1', 'finetune', m1_model, train_loader,
                                       val_df, eval_transform, median_age, cfg, device,
                                       init_ckpt=b1_dir / 'best.pt')

    # Evaluasi Menyeluruh M1 Checkpoint Terbaik
    best_m1_ckpt = torch.load(m1_dir / 'best.pt', map_location='cpu')
    eval_model = build_model(median_age, dropout=args.dropout)
    eval_model.load_state_dict(best_m1_ckpt['model'])
    eval_model.to(device)

    preds = predict_ages(eval_model, val_df, eval_transform, args.batch_size, args.num_workers, device, tta_flip=True)
    labels = val_df.age.to_numpy(dtype=np.float64)
    err = preds - labels

    metrics = {
        'run_tag': args.run_tag,
        'seed': args.seed,
        'MAE': round(float(np.mean(np.abs(err))), 3),
        'MSE': round(float(np.mean(err ** 2)), 2),
        'RMSE': round(float(np.sqrt(np.mean(err ** 2))), 3),
        'MedAE': round(float(np.median(np.abs(err))), 3),
        'bias': round(float(np.mean(err)), 3),
        'R2': round(float(1.0 - np.mean(err ** 2) / np.var(labels)), 3),
        'B0_MAE': round(float(np.mean(np.abs(labels - median_age))), 3),
    }
    metrics['reduction_vs_B0_pct'] = round(float((1.0 - metrics['MAE'] / metrics['B0_MAE']) * 100), 1)

    # Metrik Age Gating
    gating = calculate_gating_metrics(labels, preds)
    metrics.update(gating)

    # Simpan hasil evaluasi
    result_file = RESULTS_DIR / f'eval_{args.run_tag}.json'
    with open(result_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n================ HASIL RUN ================')
    print(f"Run Tag     : {args.run_tag}")
    print(f"M1 Val MAE  : {metrics['MAE']} tahun (B0 baseline: {metrics['B0_MAE']} thn, -{metrics['reduction_vs_B0_pct']}%)")
    print(f"M1 RMSE     : {metrics['RMSE']}")
    print(f"M1 MedAE    : {metrics['MedAE']}")
    print(f"Bias        : {metrics['bias']}")
    print(f"L=18 T_rec  : {metrics['L18_T_rec']} (FPR: {metrics['L18_FPR']}%, FNR: {metrics['L18_FNR']}%)")
    print(f"L=21 T_rec  : {metrics['L21_T_rec']} (FPR: {metrics['L21_FPR']}%, FNR: {metrics['L21_FNR']}%)")
    print(f'Hasil disimpan di {result_file}')
    print('============================================\n')

if __name__ == '__main__':
    main()
