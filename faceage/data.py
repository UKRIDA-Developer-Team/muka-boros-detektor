"""Audit UTKFace dan manifest split berbasis kelompok duplikat."""
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .io import read_csv, write_csv, write_json
from .metrics import AGE_GROUPS, age_group

SPLITS = ('train','validation','test')
FIELDS = ['path','age','age_group','width','height','byte_sha256','pixel_sha256','dhash','group_id','split']


def parse_age(filename):
    match = re.match(r'^(\d+)_\d+_\d+_.+\.(?:jpg|jpeg|png)$', Path(filename).name, re.IGNORECASE)
    if not match or not 0 <= int(match[1]) <= 116:
        raise ValueError('Nama berkas UTKFace atau label usia tidak valid.')
    return int(match[1])


def pixel_hash(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def dhash(image):
    values = np.asarray(image.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits = (values[:,1:] > values[:,:-1]).ravel()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


class _UnionFind:
    def __init__(self, count): self.parent = list(range(count))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, left, right):
        self.parent[self.find(right)] = self.find(left)


class _HashTree:
    """BK tree untuk pencarian Hamming tanpa membandingkan semua pasangan."""
    def __init__(self): self.root = None

    def nearby(self, value, radius):
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius: yield node[1],distance
            stack.extend(child for delta,child in node[2].items()
                         if distance-radius <= delta <= distance+radius)

    def add(self, value, index):
        if self.root is None:
            self.root = [value,index,{}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0: return
            if distance not in node[2]:
                node[2][distance] = [value,index,{}]
                return
            node = node[2][distance]


def audit_images(root, near_threshold=4):
    root = Path(root).resolve()
    if not root.is_dir(): raise ValueError(f'Folder dataset tidak ditemukan: {root}')
    if near_threshold is not None and not 0 <= near_threshold <= 8:
        raise ValueError('Threshold kemiripan harus 0-8 atau dinonaktifkan.')
    rows, invalid, near = [], [], []
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    for path in paths:
        name = path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root): raise ValueError('Symlink keluar folder data.')
            age = parse_age(path.name)
            with Image.open(path) as original:
                original.load()
                image = ImageOps.exif_transpose(original).convert('RGB')
            rows.append({'path':name,'age':age,'age_group':age_group(age),
                         'width':image.width,'height':image.height,
                         'byte_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'pixel_sha256':pixel_hash(image), 'dhash':f'{dhash(image):016x}',
                         '_texture':float(np.std(np.asarray(image.convert('L'))))})
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            invalid.append({'path':name,'reason':str(exc)})
    if not rows: raise ValueError('Tidak ada citra UTKFace valid. Periksa folder dan nama berkas.')
    groups = _UnionFind(len(rows))
    exact, tree = {}, _HashTree()
    for index,row in enumerate(rows):
        pixel = row['pixel_sha256']
        if pixel in exact: groups.join(index,exact[pixel])
        else: exact[pixel] = index
        if near_threshold is not None and row['_texture'] >= 4:
            value = int(row['dhash'],16)
            for other,distance in tree.nearby(value,near_threshold):
                if rows[other]['pixel_sha256'] != pixel:
                    near.append({'left':rows[other]['path'],'right':row['path'],'hamming_distance':distance})
                groups.join(index,other)
            tree.add(value,index)
    keys = defaultdict(list)
    for index,row in enumerate(rows): keys[groups.find(index)].append(row['pixel_sha256'])
    for index,row in enumerate(rows):
        row['group_id'] = 'g_' + min(keys[groups.find(index)])[:20]
        row.pop('_texture')
    grouped = defaultdict(list)
    for row in rows: grouped[row['group_id']].append(row)
    report = {'source':'https://susanqq.github.io/UTKFace/', 'scanned_images':len(paths),
              'valid_images':len(rows), 'invalid_images':len(invalid),
              'unique_pixel_images':len(exact), 'duplicate_groups':sum(len(v)>1 for v in grouped.values()),
              'independent_groups':len(grouped), 'near_pairs':len(near),
              'near_threshold':near_threshold,
              'conflicting_age_groups':sum(len({r['age'] for r in v})>1 for v in grouped.values()),
              'age_distribution':dict(sorted(Counter(r['age'] for r in rows).items())),
              'age_group_distribution':dict(Counter(r['age_group'] for r in rows)),
              'limitations':['Labels berasal dari estimasi DEX dan pemeriksaan manusia.',
                            'dHash mengelompokkan kemiripan konservatif; bukan verifikasi identitas.',
                            'Pemisahan berdasarkan orang yang sama tidak dapat dijamin.']}
    return rows, report, invalid, near


def assign_splits(rows, seed=42, ratios=(0.7,0.15,0.15)):
    if len(ratios)!=3 or any(r<=0 for r in ratios) or abs(sum(ratios)-1)>1e-8:
        raise ValueError('Rasio split harus tiga nilai positif dengan total 1.')
    rows = [dict(row) for row in rows]
    grouped = defaultdict(list)
    for row in sorted(rows,key=lambda r:r['path']): grouped[row['group_id']].append(row)
    if len(grouped)<3: raise ValueError('Minimal tiga kelompok independen diperlukan untuk train/validation/test.')
    buckets = defaultdict(list)
    for key, members in sorted(grouped.items()):
        buckets[age_group(statistics.median(int(r['age']) for r in members))].append((key,members))
    rng, assigned, totals = random.Random(seed), {}, Counter()
    for bucket in AGE_GROUPS:
        items = buckets[bucket]
        rng.shuffle(items)
        items.sort(key=lambda x:len(x[1]),reverse=True)
        counts = Counter()
        for key,members in items:
            split_index = min(range(3),key=lambda i:(counts[SPLITS[i]]/ratios[i],
                                                   totals[SPLITS[i]]/ratios[i],i))
            split = SPLITS[split_index]
            assigned[key] = split
            counts[split] += len(members)
            totals[split] += len(members)
    # Sparse age buckets may each contain one group; keep all three sets usable.
    for missing in SPLITS:
        if totals[missing] == 0:
            candidates = [key for key,members in grouped.items()
                          if sum(v==assigned[key] for v in assigned.values())>1]
            key = min(candidates,key=lambda k:(len(grouped[k]),k))
            totals[assigned[key]] -= len(grouped[key]); assigned[key] = missing
            totals[missing] += len(grouped[key])
    for row in rows: row['split'] = assigned[row['group_id']]
    rows.sort(key=lambda r:r['path'])
    validate_manifest(rows)
    return rows


def validate_manifest(rows):
    if not rows: raise ValueError('Manifest kosong.')
    seen, groups, pixels = set(), {}, {}
    for row in rows:
        name = row['path'].replace('\\','/')
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or not name:
            raise ValueError('Path manifest harus relatif dan tetap di dalam folder data.')
        if name in seen: raise ValueError('Path citra ganda dalam manifest.')
        seen.add(name)
        age = int(row['age'])
        if str(age)!=str(row['age']) or not 0<=age<=116: raise ValueError('Usia manifest tidak valid.')
        split = row['split']
        if split not in SPLITS: raise ValueError('Nama split tidak dikenal.')
        for container,key in [(groups,row['group_id']), (pixels,row.get('pixel_sha256'))]:
            if key:
                if key in container and container[key]!=split:
                    raise ValueError('Kebocoran kelompok duplikat melintasi split.')
                container[key] = split


def load_manifest(path):
    rows = read_csv(path)
    for row in rows: row['age'] = int(row['age'])
    validate_manifest(rows)
    return rows


def manifest_fingerprint(rows):
    validate_manifest(rows)
    canonical = [{k:r[k] for k in ['path','age','group_id','split','pixel_sha256']} for r in sorted(rows,key=lambda r:r['path'])]
    return hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()


def prepare_data(root, output, seed=42, near_threshold=4, force=False):
    output = Path(output)
    if (output/'manifest.csv').exists() and not force:
        raise ValueError('Manifest sudah ada; gunakan split ini atau --force untuk audit baru sebelum eksperimen.')
    rows,report,invalid,near = audit_images(root,near_threshold)
    manifest = assign_splits(rows,seed)
    report.update({'seed':seed, 'requested_ratios':dict(zip(SPLITS,[0.7,0.15,0.15])),
                   'split_counts':dict(Counter(r['split'] for r in manifest)),
                   'fingerprint':manifest_fingerprint(manifest),
                   'split_age_groups':{s:dict(Counter(age_group(r['age']) for r in manifest if r['split']==s)) for s in SPLITS}})
    write_csv(output/'inventory.csv',rows,FIELDS[:-1])
    write_csv(output/'manifest.csv',manifest,FIELDS)
    write_csv(output/'invalid_files.csv',invalid,['path','reason'])
    write_csv(output/'near_duplicates.csv',near,['left','right','hamming_distance'])
    count = Counter(r['group_id'] for r in manifest)
    write_csv(output/'duplicate_groups.csv',[r for r in manifest if count[r['group_id']]>1],FIELDS)
    write_json(output/'audit.json',report)
    return report


def verify_data_files(root, rows):
    root = Path(root).resolve()
    for row in rows:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Path citra keluar folder data.')
        if not path.is_file(): raise ValueError(f'Citra tidak ditemukan: {row["path"]}')
        if row.get('byte_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=row['byte_sha256']:
            raise ValueError(f'Citra berubah setelah audit: {row["path"]}. Jalankan audit baru sebelum eksperimen.')
