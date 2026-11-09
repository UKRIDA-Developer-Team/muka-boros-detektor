---
name: bandingin-run
description: Bandingkan hasil evaluasi notebook age-gating dari dua commit (MAE, MSE, RMSE, R², MAE per kelompok usia, age-gating, log epoch), lalu tambahkan tabel perbandingan ke comparison/comparison_results.md. Pakai saat user minta "bandingkan run", "bandingin commit A dan B", atau "hasil run terbaru vs sebelumnya".
---

# Tujuan

Membandingkan output yang **tersimpan** di notebook `AgeGating_MobileNetV3Small_UTKFace.ipynb` pada dua commit. Model tidak dijalankan ulang; semua angka diambil dari output sel.

# Langkah

## 1. Tentukan commit A (lama) dan commit B (baru)

- Bila user menyebut commit, pakai itu.
- Bila tidak:
  - jalankan `git log --oneline -- AgeGating_MobileNetV3Small_UTKFace.ipynb`;
  - commit B = commit terbaru yang notebook-nya punya output hasil run;
  - commit A = commit sebelumnya yang outputnya berasal dari `RUN_TAG` yang berbeda.
- Bila masih ragu, tanyakan ke user.

## 2. Ambil notebook dari kedua commit

Jalankan `git show <commit>:AgeGating_MobileNetV3Small_UTKFace.ipynb` dan simpan hasilnya di direktori sementara (scratchpad), bukan di repo. Baca JSON-nya dengan Python.

Setiap sel kode diawali penanda `# @cell: <nama>`. Data yang dibutuhkan ada di output sel berikut:

| Sel | Isi |
|---|---|
| `config` (sumber, bukan output) | `CFG`: `RUN_TAG`, `BATCH_SIZE`, LR, `MAX_EPOCHS`, dll. |
| `env` | GPU yang dipakai |
| `audit` | `SHA-256 manifest: ...` |
| `unlock` | `Evaluasi pada split: val (n = ...), seed ...` |
| `train` | log `[M1_s43_r3] epoch N \| ... MAE latih X \| MAE val Y \| Z s`, `early stopping`, `sudah selesai, dipakai ulang` |
| `curves` | tabel `best_val_mae`, `best_epoch`, `epochs_run`, `train_minutes` |
| `metrics` | tabel MAE, RMSE, MedAE, bias, penurunan MAE vs B0 per metode |
| `groups` | MAE per kelompok usia, jenis kelamin, ras |
| `gating` | FPR, FNR, AUC per L dan T; kolom `T rekomendasi (val)` berisi `ya` (versi lama: `<-`) = T rekomendasi |
| `bootstrap` | selisih MAE berpasangan beserta CI |
| `efficiency` | latensi CPU dan jumlah parameter |
| `tta` | M1 dengan dan tanpa TTA flip: MAE, T rekomendasi, FNR, AUC, latensi, dan selisih MAE bootstrap (sel baru; tidak ada di commit lama) |
| `eyes` | persentase citra yang posisi matanya terdeteksi (sel baru sejak r5) |
| `glasses_eval` | kepekaan terhadap kacamata buatan per model: rerata dan median selisih prediksi, porsi selisih > 5 th, MAE tanpa dan dengan kacamata, selisih per kelompok usia. Lebih kecil (mendekati 0) lebih baik. Baris `M1 r4` dan sejenisnya adalah run lama pembanding (sel baru sejak r5) |

Tabel DataFrame paling akurat dibaca dari `data["text/html"]`. Versi `text/plain` sering terpotong atau terbungkus ke beberapa baris.

## 3. Cek kelayakan sebelum membandingkan

Tulis setiap temuan berikut di bagian **Peringatan**:

- **Output basi:** `RUN_TAG` di `config` tidak sama dengan tag run di log `train`. Artinya notebook diedit setelah dijalankan.
- **Ada sel error**, yaitu output dengan `output_type == "error"`.
- **Split, manifest, atau seed berbeda.** Bila split atau manifest berbeda, angkanya tidak setara.
- **GPU atau CPU berbeda.** Ini hanya memengaruhi waktu dan latensi, bukan akurasi.
- **Notebook tanpa output.** Bila ini terjadi, hentikan proses dan beri tahu user.

## 4. Hitung metrik

- **Ambil apa adanya dari output:** MAE, RMSE, MedAE, bias, penurunan vs B0, MAE per kelompok, T rekomendasi, FPR/FNR di T rekomendasi, AUC, selisih bootstrap.
- **Turunan** (tandai "turunan" di kolom Keterangan):
  - MSE = RMSE².
  - R² = 1 − MSE / Var(usia), dengan Var(usia) = RMSE_B0² − bias_B0².
  - Rumus Var(usia) berlaku karena B0 selalu menebak satu angka konstan.
- **Log epoch, per model (B1, M1, B1c):**
  - ambil epoch terbaik, jumlah epoch yang dijalankan, MAE val terbaik (dari tabel `curves`), dan MAE latih di epoch terbaik (dari log);
  - hitung selisih val − latih sebagai indikator *overfitting*;
  - bila epoch terbaik = epoch terakhir, catat "cek apakah MAE val masih turun".
- **Konfigurasi:** daftar semua key `CFG` yang nilainya berbeda.

Arah "lebih baik":

| Metrik | Lebih baik bila |
|---|---|
| MAE, MSE, RMSE, MedAE, FNR, selisih val − latih, latensi | lebih kecil |
| \|bias\| | lebih dekat ke 0 |
| R², AUC, penurunan vs B0 | lebih besar |
| T rekomendasi | lebih dekat ke L |
| FPR | tidak dinilai, karena sudah dibatasi ≤ target |

## 5. Tulis ke `docs/comparison/comparison_results.md`

- Bila file belum ada, buat dengan judul `# Hasil Perbandingan Run`.
- Bila sudah ada, **tambahkan** bagian baru di akhir. Jangan menimpa isi lama.

Struktur bagian:

```markdown
## <tanggal commit B> · `<sha A>` vs `<sha B>`

- **Commit A:** `<sha>` <pesan> (<author>, <tanggal>) · run <tag> · split <split> · seed <seed> · GPU <gpu>
- **Commit B:** ...

### Ringkasan
3–5 poin, bahasa sederhana: angka utama (MAE M1), perubahan terbesar per kelompok usia,
dampak ke age-gating, dan tanda overfitting/underfitting. Dugaan penyebab hanya boleh
dari tabel "Konfigurasi yang berubah", dan ditulis sebagai "kemungkinan karena ...".

### Peringatan
### Konfigurasi yang berubah
### Metrik utama            (M1, B1 control, B1)
### MAE M1 per kelompok     (usia, jenis kelamin, ras)
### Age-gating              (M1 dan B1 control, tiap L)
### Log pelatihan
### Efisiensi
```

Semua tabel memakai format berikut:

```markdown
| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | M1 · MAE (tahun) | 6.716 | 5.658 | +1.058 (+15.8%) | B lebih baik | David |
| 2026-11-02 | M1 · MSE | 88.00 | 64.69 | +23.31 (+26.5%) | B lebih baik; turunan | David |
| 2026-11-02 | M1 · RMSE | 9.381 | 8.043 | +1.338 (+14.3%) | B lebih baik | David |
| 2026-11-02 | M1 · R² | 0.769 | 0.830 | +0.061 (+8.0%) | B lebih baik; turunan | David |
| 2026-11-02 | M1 · L=18 · FNR di T rek. | 46.0% | 11.4% | +34.6 poin | B lebih baik; dewasa yang tertahan | David |
```

Aturan kolom:

- **Date** dan **Author** diisi dari commit B.
- **Performance Improvement** bernilai positif bila B lebih baik, sesuai tabel arah di atas.
  - Metrik biasa: selisih plus persentase relatif.
  - Metrik persen (FPR, FNR, penurunan vs B0): selisih dalam "poin".
  - Baris konfigurasi: isi `-`.
- **Keterangan:** "B lebih baik", "A lebih baik", "sama", atau "tidak ada di commit X", ditambah catatan singkat bila perlu.
- Bilangan bulat ditulis tanpa pemisah ribuan. Contoh: `n=1516`, bukan `n=1.516`.

## 6. Laporkan ke user

Sebutkan path file dan isi Ringkasan, ditambah peringatan bila ada. Jangan menyalin semua tabel ke chat.

# Yang tidak boleh dilakukan

1. Jangan menambahkan informasi yang tidak terkait dengan perbandingan kedua commit.
2. Jangan memakai bahasa yang terlalu teknis. Bila istilah teknis perlu, jelaskan singkat (misalnya "FNR = dewasa yang tertahan").
3. Jangan menambahkan opini atau asumsi yang tidak didukung data. Dugaan penyebab hanya boleh berdasarkan perubahan konfigurasi.
4. Jangan mengubah format tabel.
5. Jangan menulis angka yang tidak ada di output notebook. Angka turunan wajib ditandai "turunan". Metrik yang tidak tersedia ditulis "tidak ada di commit X", jangan ditebak.
6. Jangan membandingkan tanpa peringatan bila split, manifest, atau `RUN_TAG` tidak cocok.
7. Jangan membuka data uji, menjalankan ulang notebook, atau mengubah file notebook.
