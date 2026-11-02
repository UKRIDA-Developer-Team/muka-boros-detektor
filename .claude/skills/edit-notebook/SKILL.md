---
name: edit-notebook
description: Edit file .ipynb (terutama AgeGating_MobileNetV3Small_UTKFace.ipynb) dengan aman tanpa mengubah format JSON Colab, supaya diff git hanya berisi baris yang memang diubah. Pakai setiap kali perlu mengubah kode atau markdown di dalam notebook.
---

# Tujuan

Mengubah isi sel notebook tanpa merusak format. Colab menyimpan notebook sebagai JSON dengan `indent=2`, karakter non-ASCII apa adanya (`ensure_ascii=False`), dan tanpa newline di akhir file. Kalau notebook ditulis ulang dengan format berbeda, diff git akan kotor ribuan baris walaupun yang diubah cuma satu baris.

Jangan pakai `Edit` atau `sed` langsung pada file `.ipynb`. Kode di dalam notebook tersimpan sebagai string JSON (kutip di-escape, `\n` di akhir tiap baris), jadi penggantian teks mentah mudah salah.

# Langkah

## 1. Periksa kondisi awal

- Jalankan `git status --short <notebook>`. Bila notebook punya perubahan yang belum di-commit, tanyakan ke user apakah perubahan itu boleh ikut terbawa.
- Pastikan format bisa dibaca dan ditulis ulang tanpa berubah:

```python
import json
s = open(PATH, encoding='utf-8').read()
nb = json.loads(s)
assert json.dumps(nb, indent=2, ensure_ascii=False) == s, 'format beda, cek indent/ensure_ascii/newline akhir'
```

Bila `assert` gagal, cari kombinasi format yang cocok (misalnya `indent=1`, atau ditambah `'\n'` di akhir), lalu pakai kombinasi itu saat menulis. Jangan lanjut sebelum format cocok.

## 2. Cari sel yang akan diubah

- Sel kode diawali penanda `# @cell: <nama>` (misalnya `config`, `data`, `model`, `train_fn`). Cari sel lewat penanda ini, bukan lewat nomor urut.
- Sel markdown dicari lewat judulnya (misalnya `## 4. Model dan prosedur pelatihan`).
- Penanda harus cocok dengan **tepat satu** sel.

## 3. Ubah dengan penggantian teks yang persis

Tulis script sementara di scratchpad (bukan di repo) dengan pola berikut:

```python
import json, sys
path = sys.argv[1]
nb = json.loads(open(path, encoding='utf-8').read())

def find(tag):
    hits = [c for c in nb['cells'] if tag in ''.join(c['source'])]
    assert len(hits) == 1, (tag, len(hits))
    return hits[0]

def sub(cell, old, new):
    s = ''.join(cell['source'])
    assert s.count(old) == 1, f'teks lama harus muncul tepat 1 kali: {old!r}'
    lines = s.replace(old, new).split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

cfg = find('# @cell: config')
sub(cfg, "    'RUN_TAG': 'r3',", "    'RUN_TAG': 'r4',")

open(path, 'w', encoding='utf-8').write(json.dumps(nb, indent=2, ensure_ascii=False))
```

Aturan:

- `old` harus disalin persis dari isi sel, termasuk indentasi. Bila `assert` gagal, baca ulang sel itu; jangan melonggarkan pencocokan.
- `source` tetap berupa list baris yang masing-masing diakhiri `\n`, kecuali baris terakhir.
- Jangan menyentuh `outputs`, `execution_count`, `metadata`, atau sel lain.
- Satu script boleh berisi beberapa `sub`. Semua `assert` dicek sebelum file ditulis, jadi bila satu gagal, notebook tidak berubah sama sekali.

## 4. Verifikasi

1. **Sintaks semua sel kode.** Baris magic (`!pip`, `%time`) dilewati:

```python
import json, ast
nb = json.load(open(PATH, encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse('\n'.join(l for l in ''.join(c['source']).splitlines()
                            if not l.lstrip().startswith(('!', '%'))))
print('sintaks OK')
```

2. **Diff bersih.** Jalankan `git diff --stat <notebook>`; jumlah baris yang berubah harus sebanding dengan editnya. Lalu periksa isinya dengan `git diff <notebook> | grep '^[-+]' | grep -v '^+++\|^---'`. Bila ada perubahan di luar yang dimaksud, misalnya seluruh file berubah, kembalikan file dengan `git checkout -- <notebook>` dan ulangi dari langkah 1.
3. **Jalankan logika baru bila bisa.** Bila perubahan menyentuh kode PyTorch, coba potongan kodenya di CPU dengan data dummy (misalnya `uv run --with torch --with torchvision python ...` dari scratchpad). Bila tidak bisa dijalankan, sampaikan ke user bahwa perubahan hanya dicek sintaksnya.

## 5. Khusus notebook proyek ini

- Bila mengubah apa pun yang memengaruhi pelatihan (hiperparameter, augmentasi, model, loss), naikkan `RUN_TAG` di sel `config`. Tanpa itu, run lama di Drive akan dipakai ulang.
- Sesuaikan juga teks markdown yang menjelaskan nilai yang diubah, misalnya bagian `## 3. Praproses` dan `## 4. Model dan prosedur pelatihan`.
- Key baru di `CFG` yang memengaruhi pelatihan juga ditambahkan ke daftar `'cfg'` di `config.json` dalam sel `train_fn`, supaya tercatat per run.
- Output sel lama dibiarkan. Output itu akan tertimpa saat notebook dijalankan ulang di Colab.

## 6. Laporkan ke user

Sebutkan sel yang diubah, isi `git diff --stat`, hasil cek sintaks, dan apa saja yang belum bisa dicek, misalnya pelatihan di GPU.

# Yang tidak boleh dilakukan

1. Jangan menulis ulang notebook dengan format JSON lain (indent berbeda, `ensure_ascii=True`, atau newline akhir yang berubah).
2. Jangan memakai `Edit`, `Write`, `sed`, atau `nbformat` untuk mengubah file `.ipynb` secara langsung.
3. Jangan menghapus atau mengosongkan output sel, kecuali user memintanya.
4. Jangan menyimpan script edit sementara di dalam repo.
5. Jangan commit kecuali user memintanya.
