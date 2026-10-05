# Notebook mandiri untuk Google Colab

Gunakan `Estimasi_Usia_Wajah_Colab.ipynb`. Seluruh kode audit, split, model, training, evaluasi, grafik, dan inferensi tertulis di notebook. Tidak perlu clone repo atau mengunggah `ComVis_Colab_Source.zip`. Notebook Windows tetap tersedia secara terpisah.

## 1. Unggah satu ZIP dataset ke Drive

Buat folder `My Drive/ComVis/` dan unggah ZIP [UTKFace Aligned & Cropped dari Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped) dengan nama `dataset.zip`. Pertahankan nama asli gambar; folder di dalam ZIP boleh dipertahankan. Ketentuan penggunaan mengikuti [pengelola UTKFace](https://susanqq.github.io/UTKFace/).

Tidak perlu mengekstrak puluhan ribu gambar ke Drive. Notebook menyalin ZIP dan mengekstraknya ke disk runtime.

## 2. Buka notebook dan pilih GPU

[Buka notebook langsung di Colab](https://colab.research.google.com/github/UKRIDA-Developer-Team/muka-boros-detektor/blob/main/Estimasi_Usia_Wajah_Colab.ipynb), atau buka [Google Colab](https://colab.research.google.com/) dan pilih File > Upload notebook untuk mengunggah file notebook. Simpan salinannya ke Drive.

Pilih Runtime > Change runtime type > GPU bila tersedia. Sel A memasang NumPy, Pillow, dan Matplotlib langsung. PyTorch/torchvision menggunakan instalasi bawaan runtime; bobot ImageNet diunduh otomatis saat B1 dimulai. Jika import torchvision atau CUDA gagal, gunakan runtime baru.

Jalankan sel B dan setujui sambungan Google Drive. Lokasi default:

```python
DRIVE_PROJECT = Path('/content/drive/MyDrive/ComVis')
DATASET_ZIP = DRIVE_PROJECT / 'dataset.zip'
```

Sesuaikan path bila lokasi ZIP berbeda. Sel C–H memuat seluruh fungsi dan mengekstrak dataset. Tidak ada file modul/config proyek yang perlu disediakan.

## 3. Jalankan training

Sel **1. Konfigurasi eksperimen** langsung mengaktifkan:

```python
DATA_VERSION = 'utkface_v1'
RUN_PREFIX = 'colab01'
RUN_PREPARE = True
RUN_TRAINING = True
RUN_COMPARE = True
RUN_FINAL_TEST = False
```

Jalankan sel berurutan. Sel 2 mengaudit data dan membuat manifest, sel 3–6 menjalankan B0/B1/M1/B1 control, dan sel 7–8 menampilkan perbandingan validasi serta grafik. M1 dan B1 control memakai checkpoint B1 yang sama. Default maksimum 30 epoch per tahap, batch 32, dan early stopping patience 5.

Untuk uji jalan singkat pada UTKFace, ubah `CONFIG['epochs'] = 2` sebelum training dan gunakan nama khusus seperti `RUN_PREFIX = 'uji02'`. Untuk eksperimen penuh, gunakan nama baru dan 30 epoch. Jika memori GPU tidak cukup, kurangi batch size sebelum memulai run baru. Jangan mengganti konfigurasi di tengah satu nama run.

## 4. Lokasi hasil dan penggunaan ulang

```text
My Drive/ComVis/
  dataset.zip
  logbook_kosong.csv
  processed/
    utkface_v1/              audit, manifest, distribusi usia
  runs/
    colab01_b0/
    colab01_b1/
    colab01_m1/
    colab01_b1_control/
    colab01_comparison/
```

Folder run berisi checkpoint (`best.pt` atau `b0.json`), `run.json`, konfigurasi, history, serta folder evaluasi dengan metrik, CSV prediksi, dan grafik. Folder comparison memuat ringkasan semua metode.

Disk runtime sementara. Setelah sesi baru, jalankan setup, ekstraksi, definisi fungsi, konfigurasi, dan audit lagi. Manifest dengan hash ZIP, seed, dan threshold yang sama digunakan kembali. Tahap selesai dapat dilewati setelah checkpoint/data diperiksa; training terputus belum mendukung resume epoch/optimizer dan perlu `RUN_PREFIX` baru. Checkpoint terbaik yang sudah disimpan tetap ada di folder lama.

Jika mengganti ZIP, seed, atau threshold audit, gunakan `DATA_VERSION` dan `RUN_PREFIX` baru agar hasil lama tetap tersedia.

## 5. Evaluasi test dan prediksi foto

Pilih model/konfigurasi dari skor **validasi**. Setelah keputusan dibekukan, set `RUN_TRAINING = False`, `RUN_COMPARE = False`, dan `RUN_FINAL_TEST = True`, jalankan ulang konfigurasi, lalu sel 9. Jangan memilih konfigurasi berdasarkan skor test.

Untuk prediksi, pilih `SELECTED_METHOD` pada konfigurasi dan set `RUN_PHOTO_UPLOAD = True` di sel 10. Unggah satu JPG/PNG wajah yang sudah dipotong. Foto diproses pada runtime dan tidak disalin ke Drive oleh sel, tetapi gambar/output dapat tersimpan bersama notebook. Prediksi memerlukan model yang sudah dilatih pada UTKFace; checkpoint sintetis ditolak untuk estimasi usia wajah.

Sel 11 membuat `logbook_kosong.csv` dengan 13 baris kosong di Drive. Isi sendiri berdasarkan pekerjaan nyata. File yang sudah ada tidak ditimpa.

## Batas verifikasi

Tes membawa **hanya salinan notebook** ke folder terpisah dan mengeksekusi sel pada proses Python terisolasi. Data tes berupa ZIP 30 gambar sintetis. Audit, empat metode satu epoch, perbandingan, penggunaan ulang hasil, penguncian evaluasi test, grafik, inferensi, dan logbook kosong diperiksa. Autentikasi Drive disimulasikan dan instalasi paket dilewati karena pustaka sudah tersedia di lingkungan tes.

Belum ada pengukuran akurasi UTKFace atau pengujian mount Drive/GPU pada akun Colab pengguna. [FAQ resmi Colab](https://research.google.com/colaboratory/faq.html) menjelaskan runtime sementara dan GPU yang bergantung pada ketersediaan. Jalankan eksperimen pada dataset asli untuk memperoleh MAE/RMSE penelitian.
