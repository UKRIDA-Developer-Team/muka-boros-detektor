# Menjalankan seluruh eksperimen di Google Colab

Gunakan `Estimasi_Usia_Wajah_Colab.ipynb`. Notebook lokal tetap tersedia untuk Windows. Versi Colab memasang dependensi pendukung, membaca arsip kode/dataset dari Drive, memakai GPU jika tersedia, dan menyimpan hasil penelitian ke Drive. Inferensi foto memakai sel notebook, tanpa menjalankan layanan Streamlit di runtime.

## 1. Siapkan dua arsip di Google Drive

Buat folder `My Drive/ComVis/`, lalu unggah:

```text
My Drive/ComVis/
  ComVis_Colab_Source.zip
  dataset.zip
```

`ComVis_Colab_Source.zip` disiapkan dari repo dan tersedia di `artifacts/local/`. Arsip hanya berisi kode, notebook, konfigurasi, tes, dan dokumentasi; dataset, model, `.venv`, serta berkas proposal tidak ikut. Untuk memperbaruinya setelah mengubah kode:

```powershell
.\.venv\Scripts\python.exe scripts/package_colab.py
```

`dataset.zip` adalah ZIP [UTKFace Aligned & Cropped dari Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped). Ubah nama ZIP unduhan menjadi `dataset.zip`; jangan ubah nama gambar di dalamnya. Tidak perlu mengekstrak puluhan ribu gambar ke Drive. Sumber ilmiah dan batas penggunaan tetap tercatat di `DATASET.md`.

## 2. Buka notebook dan pilih GPU

Buka [Google Colab](https://colab.research.google.com/), pilih File > Upload notebook, lalu unggah `Estimasi_Usia_Wajah_Colab.ipynb`. Simpan salinannya ke Drive. Pilih Runtime > Change runtime type > GPU (misalnya T4 jika tersedia).

Jalankan sel A untuk menyambungkan Drive dan mengizinkan akses melalui dialog Colab. Sel B menyalin arsip ke disk runtime dan mengekstraknya. Sel C memasang dependensi pendukung dan menampilkan status GPU. Jangan memasang `requirements-lock-cpu.txt` di Colab GPU; notebook menggunakan torch/torchvision bawaan runtime.

## 3. Jalankan eksperimen

Pada sel konfigurasi, pelaksanaan pertama menggunakan:

```python
RUN_PREFIX = 'colab01'
RUN_PREPARE = True
RUN_TRAINING = True
RUN_COMPARE = True
RUN_FINAL_TEST = False
RUN_SMOKE = False
```

Jalankan dari atas ke bawah. Tahap B0, B1, M1, dan B1 control dapat dijalankan per sel. M1/B1 control membutuhkan checkpoint B1. Konfigurasi default maksimal 30 epoch per tahap, batch 32, early stopping patience 5. Jika ingin uji jalan singkat, pilih nama run khusus dan ubah `CONFIG['epochs'] = 2` pada sel konfigurasi sebelum training; hasilnya belum menggantikan eksperimen penuh. Jangan mengganti konfigurasi di tengah satu nama run.

## 4. Hasil tetap tersimpan

```text
My Drive/ComVis/
  processed/             manifest dan audit
  runs/
    colab01_b0/
    colab01_b1/
    colab01_m1/
    colab01_b1_control/
    colab01_comparison/
```

Disk `/content` sementara. Setelah runtime berakhir, jalankan bootstrap lagi untuk menyalin dan mengekstrak arsip. Manifest di Drive digunakan kembali. Tahap yang telah selesai dengan konfigurasi/split sama dapat dilewati otomatis; checkpoint tersimpan diperiksa keberadaannya. Tahap yang terputus belum mendukung resume epoch atau optimizer: pilih nama run baru untuk mengulang. Checkpoint terbaik yang sudah sempat tersimpan tetap ada di folder run lama.

Jika mengganti arsip dataset atau seed, pilih folder `PROCESSED` baru dan `RUN_PREFIX` baru. Perbandingan harus memakai manifest yang sama; kode memeriksa fingerprint dan hash citra.

## 5. Prediksi dan evaluasi akhir

Setelah M1 tersedia, aktifkan `RUN_PHOTO_UPLOAD` pada sel terakhir. Pilih satu foto wajah yang sudah dipotong. Foto diproses pada memori dan disk sementara Colab; sel tidak menyalinnya ke Drive. Gambar/output dapat tersimpan bersama notebook jika kamu menyimpan output.

Setelah memilih konfigurasi dari validasi, ubah flag training/prepare/compare menjadi `False`, aktifkan `RUN_FINAL_TEST`, jalankan ulang sel konfigurasi, lalu sel evaluasi akhir. Jangan memilih konfigurasi berdasarkan skor test.

Logbook tetap kosong. Salin `docs/logbook_kosong.csv` ke Sheets atau Drive, lalu isi sendiri berdasarkan pekerjaan nyata. `.gitignore` hanya memengaruhi Git; tidak menghalangi dataset/model disimpan di Drive.

## Sumber dan batas verifikasi

[FAQ resmi Colab](https://research.google.com/colaboratory/faq.html) menjelaskan bahwa GPU tidak dijamin, runtime bersifat sementara, dan pembacaan banyak file langsung dari Drive dapat bermasalah. Karena itu arsip dataset diekstrak pada disk runtime.

Bootstrap arsip serta sel penelitian diuji lokal melalui simulasi folder Drive dan data sintetis. Login/mount Drive, runtime GPU Colab, serta eksperimen UTKFace sebenarnya harus dijalankan pada akun pengguna; belum diklaim telah diuji di Colab.
