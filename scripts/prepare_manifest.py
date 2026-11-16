#!/usr/bin/env python3
"""
scripts/prepare_manifest.py
Membangun manifest dataset UTKFace dan menghitung posisi mata (YuNet).
Sesuai rancangan proposal Bab 2 dan implementasi notebook AgeGating_MobileNetV3Small_UTKFace.ipynb.
"""

import os
import sys
import re
import io
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'utkface'
MANIFEST_DIR = ROOT / 'manifest'
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = MANIFEST_DIR / 'manifest.csv'
YUNET_PATH = ROOT / 'data' / 'face_detection_yunet_2023mar.onnx'

AGE_RANGE = (0, 116)
AGE_BINS = [(0, 12), (13, 17), (18, 24), (25, 39), (40, 59), (60, 116)]
AGE_BIN_LABELS = [f'{lo}-{hi}' if hi < AGE_RANGE[1] else f'{lo}+' for lo, hi in AGE_BINS]
SPLIT_FRAC = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SPLIT_SEED = 42
DHASH_MERGE_MAX = 2
DHASH_REVIEW_MAX = 6

GENDER_NAMES = {0: 'laki-laki', 1: 'perempuan'}
RACE_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
FILENAME_LABEL_PATTERN = re.compile(r'^(\d+)_(\d+)_(\d+)_(\d+)')
BITS_SET_PER_BYTE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def age_bin(age):
    for (lo, hi), label in zip(AGE_BINS, AGE_BIN_LABELS):
        if lo <= age <= hi:
            return label
    raise ValueError(age)

def parse_label_from_filename(file_name):
    match = FILENAME_LABEL_PATTERN.match(file_name)
    if not match:
        return None, 'format nama berkas tidak sesuai'
    age, gender, race = int(match.group(1)), int(match.group(2)), int(match.group(3))
    min_age, max_age = AGE_RANGE
    if not min_age <= age <= max_age:
        return None, f'usia di luar rentang {min_age}-{max_age}'
    if gender not in GENDER_NAMES:
        return None, 'kode jenis kelamin tidak valid'
    if race not in RACE_NAMES:
        return None, 'kode ras tidak valid'
    return (age, gender, race), None

def dhash_64bit(image):
    gray = np.asarray(image.convert('L').resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes(), 'big')

def inspect_image_file(path):
    record = {'path': str(path.relative_to(DATA_DIR)), 'name': path.name}
    try:
        raw_bytes = path.read_bytes()
        record['sha256'] = hashlib.sha256(raw_bytes).hexdigest()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            image.load()
            record['mode'] = image.mode
            rgb = image.convert('RGB')
        pixels = np.asarray(rgb)
        record.update(width=rgb.width, height=rgb.height,
                      pix_md5=hashlib.md5(pixels.tobytes() + str(pixels.shape).encode()).hexdigest(),
                      dhash=dhash_64bit(rgb))
    except Exception as e:
        record['reason'] = f'citra rusak atau tidak terbaca ({type(e).__name__})'
        return record
    label, reason = parse_label_from_filename(path.name)
    if label is None:
        record['reason'] = reason
    else:
        record.update(age=label[0], gender=label[1], race=label[2])
    return record

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)

def count_set_bits(values):
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(values)
    return BITS_SET_PER_BYTE[values.view(np.uint8)].reshape(*values.shape, 8).sum(-1)

def find_similar_pairs(hashes, max_distance, block_size=512):
    hashes = np.asarray(hashes, dtype=np.uint64)
    blocks = []
    for start in range(0, len(hashes), block_size):
        distance = count_set_bits(hashes[start:start + block_size, None] ^ hashes[None, :])
        i, j = np.nonzero(distance <= max_distance)
        keep = j > i + start
        i, j = i[keep], j[keep]
        blocks.append(np.stack([i + start, j, distance[i, j]], 1))
    return np.concatenate(blocks).astype(int) if blocks else np.zeros((0, 3), int)

def assign_groups_to_splits(groups_df, split_frac, seed):
    rng = np.random.default_rng(seed)
    split_of_group = {}
    for _, stratum_groups in groups_df.groupby('stratum', sort=True):
        group_ids = stratum_groups['group'].to_numpy().copy()
        group_size = dict(zip(stratum_groups['group'], stratum_groups['size']))
        rng.shuffle(group_ids)
        stratum_total = stratum_groups['size'].sum()
        filled = {split: 0 for split in split_frac}
        for group_id in group_ids:
            split = max(split_frac, key=lambda s: split_frac[s] * stratum_total - filled[s])
            split_of_group[group_id] = split
            filled[split] += group_size[group_id]
    return split_of_group

def detect_eyes(detector, image, pad_frac=0.25):
    rgb = np.asarray(image.convert('RGB'))
    pad = int(pad_frac * max(rgb.shape[:2]))
    padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    detector.setInputSize((padded.shape[1], padded.shape[0]))
    _, faces = detector.detect(cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])
    return face[4:8] - pad

def build_manifest():
    image_files = sorted(p for p in DATA_DIR.rglob('*')
                         if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                         and not p.name.startswith('.') and '__MACOSX' not in p.parts)
    print(f'Menemukan {len(image_files)} berkas di {DATA_DIR}')
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        audit_df = pd.DataFrame(list(pool.map(inspect_image_file, image_files)))
    if 'reason' not in audit_df:
        audit_df['reason'] = np.nan
    print(f'Audit {len(audit_df)} berkas selesai dalam {time.time() - t0:.1f} s')

    excluded_parts = [audit_df[audit_df.reason.notna()]]
    valid_df = audit_df[audit_df.reason.isna()].sort_values('path').reset_index(drop=True)
    int_columns = ['age', 'gender', 'race', 'width', 'height']
    valid_df[int_columns] = valid_df[int_columns].astype(int)

    union_find = UnionFind(len(valid_df))
    for hash_column in ['sha256', 'pix_md5']:
        for same_hash_idx in valid_df.groupby(hash_column).indices.values():
            for j in same_hash_idx[1:]:
                union_find.union(same_hash_idx[0], j)
    valid_df['exact_group'] = [union_find.find(i) for i in range(len(valid_df))]
    drop_reason = pd.Series(np.nan, index=valid_df.index, dtype=object)
    for _, duplicates in valid_df.groupby('exact_group'):
        if len(duplicates) == 1:
            continue
        if duplicates.age.nunique() == 1:
            drop_reason[duplicates.index[1:]] = 'duplikat identik (disimpan satu)'
        else:
            drop_reason[duplicates.index] = 'duplikat identik dengan label usia berbeda'
    valid_df['reason'] = drop_reason
    excluded_parts.append(valid_df[valid_df.reason.notna()])
    kept_df = valid_df[valid_df.reason.isna()].drop(columns=['reason', 'exact_group']).reset_index(drop=True)

    similar_pairs = find_similar_pairs(kept_df.dhash.tolist(), DHASH_REVIEW_MAX)
    union_find = UnionFind(len(kept_df))
    for i, j, distance in similar_pairs:
        if distance <= DHASH_MERGE_MAX:
            union_find.union(i, j)
    kept_df['group'] = [union_find.find(i) for i in range(len(kept_df))]
    has_pairs = len(similar_pairs) > 0
    near_pairs_df = pd.DataFrame({'path_a': kept_df.path.values[similar_pairs[:, 0]] if has_pairs else [],
                                  'path_b': kept_df.path.values[similar_pairs[:, 1]] if has_pairs else [],
                                  'age_a': kept_df.age.values[similar_pairs[:, 0]] if has_pairs else [],
                                  'age_b': kept_df.age.values[similar_pairs[:, 1]] if has_pairs else [],
                                  'dhash_dist': similar_pairs[:, 2] if has_pairs else []})
    near_pairs_df['action'] = np.where(near_pairs_df.dhash_dist <= DHASH_MERGE_MAX,
                                       'digabung (satu split)', 'tinjau saja')
    near_pairs_df.to_csv(MANIFEST_DIR / 'near_pairs.csv', index=False)

    kept_df['age_bin'] = kept_df.age.map(age_bin)
    groups_df = kept_df.groupby('group').agg(size=('path', 'size'), med_age=('age', 'median')).reset_index()
    groups_df['stratum'] = groups_df.med_age.map(lambda a: age_bin(int(round(a))))
    kept_df['split'] = kept_df.group.map(assign_groups_to_splits(groups_df, SPLIT_FRAC, SPLIT_SEED))

    manifest_columns = ['path', 'age', 'gender', 'race', 'age_bin', 'group', 'split', 'width', 'height', 'mode',
                        'sha256', 'dhash']
    kept_df[manifest_columns].to_csv(MANIFEST_PATH, index=False)
    excluded_df = pd.concat(excluded_parts)[['path', 'reason']]
    excluded_df.to_csv(MANIFEST_DIR / 'excluded.csv', index=False)

    with open(MANIFEST_PATH, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f'Manifest dibuat: {MANIFEST_PATH} (SHA-256: {sha256})')
    print('Distribusi split:', kept_df.split.value_counts().to_dict())
    return kept_df, sha256

def compute_eyes(manifest_df, manifest_sha):
    eyes_path = MANIFEST_DIR / f'eyes_{manifest_sha[:8]}.csv'
    if eyes_path.exists():
        print('Eyes data sudah ada:', eyes_path)
        return pd.read_csv(eyes_path)
    if not YUNET_PATH.exists():
        print(f'YuNet ONNX tidak ditemukan di {YUNET_PATH}')
        return None
    print('Menghitung posisi mata dengan YuNet...')
    detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), '', (320, 320), 0.7)
    rows = []
    t0 = time.time()
    train_val_df = manifest_df[manifest_df.split.isin(['train', 'val'])].reset_index(drop=True)
    for i, row in train_val_df.iterrows():
        img_path = DATA_DIR / row['path']
        try:
            with Image.open(img_path) as img:
                eyes = detect_eyes(detector, img)
        except Exception:
            eyes = None
        if eyes is not None:
            rows.append({'path': row['path'], 'eye_right_x': eyes[0], 'eye_right_y': eyes[1],
                         'eye_left_x': eyes[2], 'eye_left_y': eyes[3]})
        else:
            rows.append({'path': row['path'], 'eye_right_x': np.nan, 'eye_right_y': np.nan,
                         'eye_left_x': np.nan, 'eye_left_y': np.nan})
        if (i + 1) % 2000 == 0 or (i + 1) == len(train_val_df):
            print(f'  {i + 1}/{len(train_val_df)} citra diproses ({time.time() - t0:.1f} s)')
    eyes_df = pd.DataFrame(rows)
    eyes_df.to_csv(eyes_path, index=False)
    print('Eyes data disimpan ke:', eyes_path)
    return eyes_df

if __name__ == '__main__':
    manifest_df, manifest_sha = build_manifest()
    compute_eyes(manifest_df, manifest_sha)
