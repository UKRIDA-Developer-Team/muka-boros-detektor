# Estimasi usia dari citra wajah

Implementasi proposal proyek visi komputer: **UTKFace + MobileNetV3 Small**, dengan regresi angka usia, baseline median, backbone beku, dan fine tuning. Repo menyediakan audit data, split tetap, pelatihan, evaluasi, grafik, inferensi satu wajah, dan demo Streamlit.

Model penelitian perlu dilatih pada UTKFace sebelum dipakai di demo. Repo tidak menyertakan wajah dataset, checkpoint terlatih, atau angka akurasi yang belum diukur. Smoke test menggunakan gambar sintetis dan tidak menghasilkan model usia wajah yang layak digunakan.

## Mulai di Windows

**Untuk Google Colab:** buka [notebook mandiri di Colab](https://colab.research.google.com/github/UKRIDA-Developer-Team/muka-boros-detektor/blob/main/Estimasi_Usia_Wajah_Colab.ipynb) atau unggah [Estimasi_Usia_Wajah_Colab.ipynb](Estimasi_Usia_Wajah_Colab.ipynb). Seluruh kode ada di notebook; cukup unggah `dataset.zip` ke `My Drive/ComVis/`, pilih GPU, lalu jalankan sel berurutan. Tidak perlu ZIP kode atau clone repo. Checkpoint dan hasil tersimpan ke Drive. Langkah lengkap ada di [panduan Colab](docs/COLAB.md).

Untuk menjalankan proyek per tahap, buka **[Estimasi_Usia_Wajah.ipynb](Estimasi_Usia_Wajah.ipynb)** di VS Code/Jupyter. Pasang `requirements-notebook.txt`, pilih kernel `.venv`, lalu ikuti sel konfigurasi. Pelatihan dan evaluasi test tidak berjalan sebelum flag terkait diaktifkan; notebook belum memuat hasil penelitian.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Gunakan Python 3.12 dan buka terminal di folder repo.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Demo tersedia di `http://localhost:8501`. Sebelum model tersedia, aplikasi menampilkan petunjuk dan tidak membuat prediksi acak. Foto unggahan diproses dalam memori sesi dan tidak ditulis ke disk.

Untuk instalasi CPU saja:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock-cpu.txt` mencatat versi lengkap yang dipakai untuk verifikasi lokal. Untuk GPU, pasang pasangan PyTorch/torchvision sesuai petunjuk [instalasi PyTorch](https://pytorch.org/get-started/locally/) terlebih dahulu. `--device auto` memakai CUDA hanya bila benar-benar tersedia; `--device cuda` gagal dengan pesan yang jelas bila tidak tersedia.

## Siapkan dataset

1. Sumber ilmiah dan ketentuan penggunaan: [halaman resmi UTKFace](https://susanqq.github.io/UTKFace/) atau [repo AICIP](https://github.com/aicip/UTKFace).
2. Untuk unduhan, gunakan [UTKFace Aligned & Cropped di Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped), salinan dari pengunggah pihak ketiga. Buka halaman, login jika diminta, lalu pilih Download. Ekstrak citra ke `data/raw/UTKFace/`. Program mencari citra secara rekursif, sehingga subfolder hasil ekstraksi boleh dipertahankan. Halaman resmi masih terbaca saat diperiksa pada 6 Oktober 2026, tetapi tautan Google Drive untuk citra cropped mengembalikan 404; URL repo lama `github.com/susanqq/UTKFace` juga mengembalikan 404.
3. Pertahankan nama asli berformat `usia_gender_race_timestamp.jpg`. Program hanya menggunakan usia sebagai target; atribut gender/race tidak menjadi input.
4. Jalankan audit sebelum membuat eksperimen.

```powershell
.\.venv\Scripts\python.exe -m faceage prepare --data data/raw/UTKFace --output data/processed
```

UTKFace ditujukan untuk penelitian nonkomersial. Pengelola menyatakan label berasal dari DEX dan pemeriksaan manusia, sehingga metrik mengukur kesesuaian terhadap label dataset. Jangan unggah ulang dataset wajah ke Git. Ketentuan dan sumber tercatat di [docs/DATASET.md](docs/DATASET.md).

Audit menghasilkan:

| Berkas | Isi |
| --- | --- |
| `inventory.csv` | Usia, dimensi, hash byte/piksel, dan hash kemiripan |
| `manifest.csv` | Daftar citra dan split train/validation/test yang tetap |
| `audit.json` | Jumlah valid/rusak, distribusi usia, ukuran tiap split, fingerprint |
| `invalid_files.csv` | Berkas yang ditolak dan alasannya |
| `duplicate_groups.csv` | Kelompok duplikat beserta usia dan split |
| `near_duplicates.csv` | Pasangan mirip untuk ditinjau |

Duplikat piksel dan kemiripan dHash dikelompokkan sebelum split. dHash adalah pencarian kemiripan, bukan pengenalan identitas; kelompoknya perlu ditinjau. Threshold awal 4 dapat dipilih sebelum eksperimen melalui `--near-threshold`. `--no-near` mematikan penggabungan kemiripan dan tetap menjaga duplikat piksel; jika digunakan, tulis alasan serta keterbatasannya. Citra nyaris polos tidak digabung hanya berdasarkan dHash.

Split menargetkan 70% latih, 15% validasi, 15% uji dengan seed 42 dan stratifikasi kelompok usia. Rasio aktual dapat berbeda karena satu kelompok tidak boleh dipisahkan. Minimal tiga kelompok independen dibutuhkan; dataset kecil atau kelompok usia langka tidak menjamin semua kelompok terwakili pada setiap split. Program tidak menjamin pemisahan identitas orang yang sama.

Manifest tidak ditimpa diam-diam. `--force` hanya digunakan untuk audit baru sebelum eksperimen; setelah itu seluruh model harus dilatih kembali pada manifest baru. Setiap run menyimpan fingerprint, dan evaluasi menolak manifest yang berbeda atau berkas citra yang berubah.

## Jalankan eksperimen proposal

```powershell
.\.venv\Scripts\python.exe -m faceage train --method b0 --data data/raw/UTKFace --output runs/b0
.\.venv\Scripts\python.exe -m faceage train --method b1 --data data/raw/UTKFace --output runs/b1
.\.venv\Scripts\python.exe -m faceage train --method m1 --data data/raw/UTKFace --output runs/m1 --init-checkpoint runs/b1/best.pt
.\.venv\Scripts\python.exe -m faceage train --method b1_control --data data/raw/UTKFace --output runs/b1_control --init-checkpoint runs/b1/best.pt
```

Atau jalankan seluruh tahap dengan skrip PowerShell:

```powershell
.\scripts\train_pipeline.ps1 -Data data/raw/UTKFace -RunPrefix experiment01 -Device auto
```

| Metode | Tujuan |
| --- | --- |
| B0 | Median label usia **data latih saja**, tanpa fitur citra |
| B1 | MobileNetV3 Small ImageNet; backbone dan statistik BatchNorm beku, head regresi dilatih |
| M1 | Dimulai dari B1; tiga modul akhir backbone serta head dibuka |
| B1 control | Dimulai dari checkpoint B1 yang sama; head dilanjutkan dengan backbone tetap beku |

B1 control dan M1 dibandingkan dengan konfigurasi serta batas epoch sama untuk menguji perubahan strategi fine tuning. Early stopping dapat membuat epoch aktual berbeda; jumlahnya tetap tercatat. B1 versus M1 juga dilaporkan sesuai proposal, tetapi tambahan tahap pelatihan perlu disebut saat menafsirkan perbedaan keduanya.

Konfigurasi di [configs/default.json](configs/default.json): input RGB 224 × 224, resize sisi pendek 256/crop tengah/normalisasi ImageNet, augmentasi latih berupa flip dan jitter ringan, AdamW, Smooth L1 beta 1 tahun, batch 32, maksimum 30 epoch per tahap, patience 5, learning rate head 0.0001 dan backbone 0.00001. Bias head awal diisi median data latih agar prediksi awal berada pada skala usia. Seluruh konfigurasi disimpan pada tiap run.

Untuk eksperimen augmentasi atau sampling, salin konfigurasi ke nama baru dan ubah satu faktor (`augmentation` atau `sampling`, dengan pilihan `natural`/`balanced`). Gunakan manifest yang sama dan nama run baru. Jangan menyetel konfigurasi berdasarkan hasil test set. Bobot ImageNet diunduh saat B1 pertama kali dibuat; checkpoint untuk evaluasi/inferensi dapat dimuat tanpa internet.

Pelatihan menghasilkan `run.json`, `config.json`, `history.csv`, `best.pt` (atau `b0.json`), dan evaluasi validasi otomatis. `run.json` mencatat hardware aktual, versi pustaka, jumlah parameter, durasi, seed, checkpoint awal, dan MAE validasi terbaik. Folder run yang sudah berisi hasil ditolak agar bukti lama tidak tertimpa.

## Evaluasi dan perbandingan

```powershell
.\.venv\Scripts\python.exe -m faceage compare --data data/raw/UTKFace --runs runs/b0 runs/b1 runs/m1 runs/b1_control --output runs/comparison
```

Default menggunakan **validation** untuk pengembangan. Sesudah model/config dipilih dan dibekukan, jalankan evaluasi akhir:

```powershell
.\.venv\Scripts\python.exe -m faceage compare --data data/raw/UTKFace --runs runs/b0 runs/b1 runs/m1 runs/b1_control --output runs/comparison --split test --final-test
```

Metrik: MAE, RMSE, median absolute error, serta jumlah citra dan MAE pada 0-12, 13-19, 20-39, 40-59, dan 60+ tahun. Kelompok kosong diberi nilai `null`, bukan skor nol. Evaluasi selalu memakai prediksi mentah, tanpa pembulatan atau pembatasan usia.

Luaran setiap `evaluation_validation/` atau `evaluation_test/`:

- `metrics.json`, `age_groups.csv`, dan `predictions.csv`;
- `best_examples.csv` dan `worst_examples.csv` untuk meninjau kasus berhasil/gagal;
- `prediction_vs_label.png`, `residual_histogram.png`, `age_group_mae.png`, dan `learning_curve.png` untuk model neural.

Contoh terbaik/terburuk menyimpan path lokal dan nilai galat, bukan salinan foto untuk publikasi. Beri kategori kegagalan dan hipotesis berdasarkan peninjauan manusia; program tidak mengarang alasan seperti pencahayaan atau pose. [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md) menyediakan formatnya.

Perbandingan menghasilkan CSV/JSON/grafik pada split yang sama, persentase perubahan MAE M1 terhadap B0, dan pemeriksaan sumber checkpoint B1 control/M1. Target penurunan 20% merupakan sasaran proposal, bukan jaminan hasil.

Waktu evaluasi rata-rata per gambar mencakup loading model/data dan batching; itu berbeda dari latency satu gambar di demo. Demo mengukur praproses dan inferensi setelah model dimuat dan dipanaskan. Jangan membandingkan kedua angka sebagai pengukuran yang sama.

## Inferensi satu foto

```powershell
.\.venv\Scripts\python.exe -m faceage predict --checkpoint runs/m1/best.pt --image path/to/cropped_face.jpg
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Gunakan satu wajah yang sudah dipotong. Program belum memiliki deteksi/crop otomatis atau pemeriksaan jumlah wajah. Demo hanya menawarkan run penelitian yang selesai; checkpoint sintetis tidak ditawarkan sebagai model usia wajah. B0 tetap dapat ditampilkan sebagai pembanding median dan diberi penjelasan bahwa citra tidak digunakan untuk prediksi.

Checkpoint berasal dari eksperimen sendiri dan dimuat dengan `weights_only=True`. Estimasi tidak menjadi dasar penentuan umur legal, identitas, atau keputusan otomatis terhadap seseorang. Validasi pada dataset lain diperlukan untuk menilai generalisasi.

## Uji teknis tanpa dataset wajah

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --output artifacts/local/smoke
.\.venv\Scripts\python.exe -m pytest -q
```

Smoke test membuat 30 gambar acak, menjalankan audit, empat metode selama satu epoch, membandingkan validasi, lalu memeriksa inferensi. Bobot backbone acak hanya boleh digunakan bersama `dataset_kind: synthetic`. Luaran smoke diberi label sintetis, tidak dihitung sebagai akurasi penelitian, dan disimpan di folder yang diabaikan Git. Jika mengulang, gunakan folder baru.

## Logbook dan panduan kuliah

Logbook sengaja **kosong**, sesuai permintaan pengguna:

- [Logbook CSV kosong](docs/logbook_kosong.csv): impor ke Google Sheets; salin per anggota.
- [Logbook Markdown kosong](docs/LOGBOOK.md): pilihan untuk mengisi langsung di repo.
- [Template checkpoint CP00-CP13](docs/checkpoint_template.csv): capaian/XP maksimum dari panduan, kolom bukti, tanggal, verifikasi, dan XP diperoleh tetap kosong.
- [Template kontribusi](docs/kontribusi_kosong.csv): nama, NIM, peran, tugas, dan bukti belum diisi.

Program tidak mengisi kontribusi atau status verifikasi dosen secara otomatis. Pemetaan artefak terhadap panduan, serta hal yang perlu disiapkan untuk sidang, ada di [docs/PANDUAN_UTS_UAS.md](docs/PANDUAN_UTS_UAS.md).

## Struktur repo

```text
app.py                      Demo Streamlit
faceage/                    Audit, model, train, evaluate, compare, predict
configs/                    Konfigurasi penelitian dan smoke
scripts/                    Pipeline PowerShell dan smoke test
tests/                      Tes data, model, metrik, CLI, dan demo
docs/                       Rujukan, panduan, analisis galat, template kosong
data/raw/                   Dataset lokal; tidak masuk Git
data/processed/             Audit dan manifest lokal; tidak masuk Git
runs/                       Checkpoint, log, metrik, grafik; tidak masuk Git
artifacts/local/            Bukti smoke lokal; tidak masuk Git
```

## Sumber dan atribusi

Lima sumber kredibel beserta relevansinya ada di [docs/REFERENCES.md](docs/REFERENCES.md), termasuk paper MobileNetV3, DEX, dan paper pengembang UTKFace. Arsitektur dan bobot ImageNet berasal dari torchvision; pustaka lain tercantum dalam dependensi. Kode pipeline ditulis untuk repo ini dengan bantuan **OpenAI Codex**, berdasarkan proposal dan panduan yang diberikan pengguna. Tidak ada klaim kode ini disalin dari implementasi paper atau menghasilkan performa paper tersebut. Mahasiswa tetap perlu memahami, menjalankan, dan menjelaskan keputusan serta hasil yang dikumpulkan.

## Status hasil

Sumber kode, tes, dan demo tersedia. Dataset UTKFace belum disertakan dan belum ada pelatihan penuh atau hasil evaluasi UTKFace yang diklaim. Laporan UAS, slide hasil, dan video demo cadangan disusun setelah eksperimen dengan data nyata selesai.
