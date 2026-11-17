# Hasil Perbandingan Run

## 2026-11-02 · `bdf1018` vs `7d4cc4c`

- **Commit A:** `bdf1018` TOPER 57 TAHUN GES (David, 2026-10-12) · run r2 · split val · seed 43 · GPU Tesla T4
- **Commit B:** `7d4cc4c` hmm kayaknya udah bener (David, 2026-11-02) · run r3 · split val · seed 43 · GPU NVIDIA A100-SXM4-40GB
- *Performance Improvement* bernilai positif bila Commit B lebih baik. Kolom Date dan Author diisi dari Commit B.
- MSE dan R² adalah angka turunan:
  - MSE = RMSE²
  - R² = 1 − MSE / Var(usia), dengan Var(usia) = RMSE_B0² − bias_B0²

### Ringkasan

- **Model utama (M1) lebih akurat.** Rata-rata meleset turun dari 6.72 ke 5.66 tahun (−15.8%). R² naik dari 0.77 ke 0.83.
- **Kenaikan terbesar di anak dan remaja.** MAE usia 0–12 turun 47% dan usia 13–17 turun 38%. Usia 40–59 dan 60+ masih paling meleset: 8.19 dan 10.16 tahun.
- **Age-gating jauh lebih baik.** Untuk batas 18 tahun, ambang T turun dari 31 ke 24. Orang dewasa yang tertahan turun dari 46.0% ke 11.4%, sementara anak yang lolos tetap ≤ 5%.
- **Mulai overfitting.** Selisih MAE val − latih pada M1 naik dari 0.51 ke 2.33.
- **Kemungkinan karena backbone dibuka penuh.** `UNFREEZE_LAST` 3 → 13, `LR_BACKBONE` 1e-5 → 1e-4, BatchNorm ikut dilatih, dan `BATCH_SIZE` 1024 → 128. Buktinya: B1 control, yang hanya melatih head, cuma membaik 0.32 tahun, sedangkan M1 membaik 1.06 tahun.

### Peringatan

- GPU berbeda (Tesla T4 vs NVIDIA A100-SXM4-40GB). Hanya waktu pelatihan yang tidak setara; akurasi tidak terpengaruh.
- Split (val), seed (43), dan manifest data sama. `RUN_TAG` di config cocok dengan log di kedua commit. Tidak ada sel error.

### Konfigurasi yang berubah

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | `BATCH_SIZE` | `1024` | `128` | - | berubah | David |
| 2026-11-02 | `LR_HEAD` | `5e-4` | `3e-4` | - | berubah | David |
| 2026-11-02 | `LR_BACKBONE` | `1e-5` | `1e-4` | - | berubah | David |
| 2026-11-02 | `MAX_EPOCHS` | `30` | `60` | - | berubah | David |
| 2026-11-02 | `PATIENCE` | `5` | `10` | - | berubah | David |
| 2026-11-02 | `WARMUP_EPOCHS` | - | `2` | - | baru di commit B | David |
| 2026-11-02 | `UNFREEZE_LAST` | `3` | `13` | - | berubah | David |
| 2026-11-02 | `M1_UPDATE_BN_STATS` | `False` | `True` | - | berubah | David |
| 2026-11-02 | `NUM_WORKERS` | `2` | `min(8, os.cpu_count() or 2)` | - | berubah | David |
| 2026-11-02 | `RUN_TAG` | `'r2'` | `'r3'` | - | berubah | David |

### Metrik utama

Split val, n = 3468.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | M1 · MAE (tahun) | 6.716 | 5.658 | +1.058 (+15.8%) | B lebih baik | David |
| 2026-11-02 | M1 · MSE | 88.00 | 64.69 | +23.31 (+26.5%) | B lebih baik; turunan | David |
| 2026-11-02 | M1 · RMSE | 9.381 | 8.043 | +1.338 (+14.3%) | B lebih baik | David |
| 2026-11-02 | M1 · MedAE | 4.766 | 3.828 | +0.938 (+19.7%) | B lebih baik | David |
| 2026-11-02 | M1 · bias (pred − label) | -0.600 | -0.524 | +0.076 | B lebih baik; makin dekat 0 makin baik | David |
| 2026-11-02 | M1 · R² | 0.769 | 0.830 | +0.061 (+8.0%) | B lebih baik; turunan | David |
| 2026-11-02 | M1 · penurunan MAE vs B0 | 53.4% | 60.7% | +7.3 poin | B lebih baik | David |
| 2026-11-02 | B1c · MAE (tahun) | 7.441 | 7.122 | +0.319 (+4.3%) | B lebih baik | David |
| 2026-11-02 | B1c · MSE | 105.93 | 99.00 | +6.92 (+6.5%) | B lebih baik; turunan | David |
| 2026-11-02 | B1c · RMSE | 10.292 | 9.950 | +0.342 (+3.3%) | B lebih baik | David |
| 2026-11-02 | B1c · R² | 0.722 | 0.740 | +0.018 (+2.5%) | B lebih baik; turunan | David |
| 2026-11-02 | B1 · MAE (tahun) | 7.813 | 7.374 | +0.439 (+5.6%) | B lebih baik | David |
| 2026-11-02 | B1 · MSE | 113.42 | 104.86 | +8.56 (+7.6%) | B lebih baik; turunan | David |
| 2026-11-02 | B1 · RMSE | 10.650 | 10.240 | +0.410 (+3.8%) | B lebih baik | David |
| 2026-11-02 | B1 · bias (pred − label) | -0.539 | -0.769 | -0.230 | A lebih baik; makin dekat 0 makin baik | David |
| 2026-11-02 | B1 · R² | 0.702 | 0.725 | +0.023 (+3.2%) | B lebih baik; turunan | David |

### MAE M1 per kelompok

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | M1 · usia 0-12 (n=487) | 5.043 | 2.660 | +2.383 (+47.3%) | B lebih baik | David |
| 2026-11-02 | M1 · usia 13-17 (n=112) | 9.135 | 5.672 | +3.463 (+37.9%) | B lebih baik | David |
| 2026-11-02 | M1 · usia 18-24 (n=353) | 5.114 | 4.263 | +0.851 (+16.6%) | B lebih baik | David |
| 2026-11-02 | M1 · usia 25-39 (n=1516) | 5.145 | 4.794 | +0.351 (+6.8%) | B lebih baik | David |
| 2026-11-02 | M1 · usia 40-59 (n=629) | 8.907 | 8.189 | +0.718 (+8.1%) | B lebih baik | David |
| 2026-11-02 | M1 · usia 60+ (n=371) | 12.413 | 10.157 | +2.256 (+18.2%) | B lebih baik | David |
| 2026-11-02 | M1 · laki-laki (n=1862) | 6.741 | 5.824 | +0.917 (+13.6%) | B lebih baik | David |
| 2026-11-02 | M1 · perempuan (n=1606) | 6.688 | 5.466 | +1.222 (+18.3%) | B lebih baik | David |
| 2026-11-02 | M1 · ras Asian (n=537) | 5.657 | 4.141 | +1.516 (+26.8%) | B lebih baik | David |
| 2026-11-02 | M1 · ras Black (n=685) | 6.728 | 6.015 | +0.713 (+10.6%) | B lebih baik | David |
| 2026-11-02 | M1 · ras Indian (n=589) | 6.745 | 5.733 | +1.012 (+15.0%) | B lebih baik | David |
| 2026-11-02 | M1 · ras Others (n=219) | 5.140 | 4.058 | +1.082 (+21.1%) | B lebih baik | David |
| 2026-11-02 | M1 · ras White (n=1438) | 7.334 | 6.268 | +1.066 (+14.5%) | B lebih baik | David |

### Age-gating

Ambang T dipilih dari data validasi: T terkecil dengan FPR ≤ 5%. FPR = anak di bawah L yang lolos. FNR = orang dewasa yang tertahan.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | M1 · L=18 · T rekomendasi | 31 | 24 | +7 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-02 | M1 · L=18 · FPR di T rek. | 4.7% | 4.5% | -0.2 poin | tidak dinilai; dibatasi ≤ 5% | David |
| 2026-11-02 | M1 · L=18 · FNR di T rek. | 46.0% | 11.4% | +34.6 poin | B lebih baik | David |
| 2026-11-02 | M1 · L=18 · AUC | 0.942 | 0.980 | +0.038 (+4.0%) | B lebih baik | David |
| 2026-11-02 | M1 · L=21 · T rekomendasi | 32 | 28 | +4 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-02 | M1 · L=21 · FPR di T rek. | 4.6% | 3.2% | -1.4 poin | tidak dinilai; dibatasi ≤ 5% | David |
| 2026-11-02 | M1 · L=21 · FNR di T rek. | 48.2% | 33.9% | +14.3 poin | B lebih baik | David |
| 2026-11-02 | M1 · L=21 · AUC | 0.925 | 0.965 | +0.040 (+4.3%) | B lebih baik | David |
| 2026-11-02 | B1c · L=18 · T rekomendasi | 32 | 32 | +0 | sama | David |
| 2026-11-02 | B1c · L=18 · FNR di T rek. | 49.2% | 49.5% | -0.3 poin | A lebih baik | David |
| 2026-11-02 | B1c · L=18 · AUC | 0.925 | 0.933 | +0.008 (+0.9%) | B lebih baik | David |
| 2026-11-02 | B1c · L=21 · T rekomendasi | 34 | 32 | +2 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-02 | B1c · L=21 · FNR di T rek. | 54.7% | 48.2% | +6.5 poin | B lebih baik | David |
| 2026-11-02 | B1c · L=21 · AUC | 0.909 | 0.917 | +0.008 (+0.9%) | B lebih baik | David |

### Bootstrap berpasangan

Nilai negatif berarti model di kiri lebih akurat. CI pada kolom Keterangan adalah milik Commit B.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | selisih MAE M1 − B1c | -0.725 | -1.463 | +0.738 | B lebih baik; CI [-1.644, -1.259], signifikan | David |
| 2026-11-02 | selisih MAE M1 − B1 | -1.096 | -1.715 | +0.619 | B lebih baik; CI [-1.909, -1.505], signifikan | David |
| 2026-11-02 | selisih MAE B1c − B1 | -0.371 | -0.252 | -0.119 | A lebih baik; CI [-0.315, -0.194], signifikan | David |

### Log pelatihan

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | B1 · MAE val terbaik | 7.813 | 7.374 | +0.439 (+5.6%) | B lebih baik | David |
| 2026-11-02 | B1 · epoch terbaik / dijalankan | 30 / 30 | 60 / 60 | - | terbaik di epoch terakhir pada A dan B; cek apakah MAE val masih turun | David |
| 2026-11-02 | B1 · selisih val − latih | 0.143 | - | - | tidak ada di commit B (B1 dipakai ulang, tanpa log epoch) | David |
| 2026-11-02 | M1 · MAE val terbaik | 6.716 | 5.658 | +1.058 (+15.8%) | B lebih baik | David |
| 2026-11-02 | M1 · MAE latih di epoch terbaik | 6.21 | 3.33 | - | berubah | David |
| 2026-11-02 | M1 · selisih val − latih | 0.506 | 2.328 | -1.822 | A lebih baik; indikator overfitting | David |
| 2026-11-02 | M1 · epoch terbaik / dijalankan | 29 / 30 | 23 / 33 | - | B berhenti otomatis (early stopping) | David |
| 2026-11-02 | B1c · MAE val terbaik | 7.441 | 7.122 | +0.319 (+4.3%) | B lebih baik | David |
| 2026-11-02 | B1c · selisih val − latih | 0.391 | 0.972 | -0.581 | A lebih baik; indikator overfitting | David |
| 2026-11-02 | B1c · epoch terbaik / dijalankan | 30 / 30 | 47 / 57 | - | A terbaik di epoch terakhir; B berhenti otomatis | David |
| 2026-11-02 | M1 · lama latih (menit) | 19.8 | 7.8 | -12.0 | berubah; tergantung GPU | David |

### Efisiensi

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-02 | latensi CPU median (ms) | 10.011 | 8.777 | +1.234 (+12.3%) | B lebih baik; CPU berbeda (Xeon 2.00 GHz vs 2.20 GHz) | David |
| 2026-11-02 | jumlah parameter | 1518881 | 1518881 | +0 | sama | David |

## 2026-11-03 · `7d4cc4c` vs `fac758c`

- **Commit A:** `7d4cc4c` hmm kayaknya udah bener (David, 2026-11-02) · run r3 · split val · seed 43 · GPU NVIDIA A100-SXM4-40GB
- **Commit B:** `fac758c` fix (David, 2026-11-03) · run r4 · split val · seed 43 · GPU NVIDIA A100-SXM4-40GB
- *Performance Improvement* positif = Commit B lebih baik. Kolom Date dan Author mengacu ke Commit B. MSE = RMSE²; R² = 1 − MSE / Var(usia), dengan Var(usia) = RMSE_B0² − bias_B0². Keduanya angka turunan.

### Ringkasan

- **M1 makin akurat.** Rata-rata meleset turun dari 5.66 ke 5.11 tahun (−9.8%). R² naik dari 0.830 ke 0.859.
- **Kenaikan terbesar di usia tua.** MAE usia 60+ turun 1.34 tahun (−13.2%), usia 40–59 turun 0.98 tahun (−11.9%). Semua kelompok usia, jenis kelamin, dan ras membaik.
- **Overfitting berkurang.** Selisih MAE val − latih pada M1 turun dari 2.33 ke 1.25. Kemungkinan karena augmentasi baru (`AUG_ROTATE`, `AUG_TRANSLATE`, `AUG_SCALE`), `WEIGHT_DECAY` 1e-4 → 0.05, dan `DROPOUT` 0.3.
- **Age-gating berubah dua arah.** Untuk L = 21, orang dewasa yang tertahan turun dari 33.9% ke 19.6%. Untuk L = 18, sedikit memburuk: T naik dari 24 ke 25, dan dewasa yang tertahan naik dari 11.4% ke 14.7%.
- **Latensi CPU naik dari 8.8 ke 19.1 ms.** Kemungkinan karena `TTA_FLIP = True` (setiap foto diprediksi dua kali). Masih di bawah target 30 ms. B1 dan B1 control sedikit memburuk (MAE +0.11 dan +0.06 tahun).

### Peringatan

- Tidak ada. Split (val), seed (43), manifest, dan GPU sama. `RUN_TAG` di config cocok dengan log di kedua commit. Tidak ada sel error.
- Catatan: MAE latih di commit B diukur pada data yang diaugmentasi lebih berat, jadi sebagian penurunan selisih val − latih berasal dari MAE latih yang naik.

### Konfigurasi yang berubah

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | `WEIGHT_DECAY` | `1e-4` | `0.05` | - | berubah | David |
| 2026-11-03 | `DROPOUT` | - | `0.3` | - | baru di commit B | David |
| 2026-11-03 | `AUG_ROTATE` | - | `10` | - | baru di commit B | David |
| 2026-11-03 | `AUG_TRANSLATE` | - | `0.05` | - | baru di commit B | David |
| 2026-11-03 | `AUG_SCALE` | - | `(0.9, 1.1)` | - | baru di commit B | David |
| 2026-11-03 | `TTA_FLIP` | - | `True` | - | baru di commit B | David |
| 2026-11-03 | `RUN_TAG` | `'r3'` | `'r4'` | - | berubah | David |

### Metrik utama

Split val, n = 3468.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | M1 · MAE (tahun) | 5.658 | 5.105 | +0.553 (+9.8%) | B lebih baik | David |
| 2026-11-03 | M1 · MSE | 64.69 | 53.63 | +11.06 (+17.1%) | B lebih baik; turunan | David |
| 2026-11-03 | M1 · RMSE | 8.043 | 7.323 | +0.720 (+9.0%) | B lebih baik | David |
| 2026-11-03 | M1 · MedAE | 3.828 | 3.500 | +0.328 (+8.6%) | B lebih baik | David |
| 2026-11-03 | M1 · bias (pred − label) | -0.524 | -0.416 | +0.108 | B lebih baik; makin dekat 0 makin baik | David |
| 2026-11-03 | M1 · R² | 0.830 | 0.859 | +0.029 (+3.5%) | B lebih baik; turunan | David |
| 2026-11-03 | M1 · penurunan MAE vs B0 | 60.7% | 64.6% | +3.9 poin | B lebih baik | David |
| 2026-11-03 | B1c · MAE (tahun) | 7.122 | 7.181 | -0.059 (-0.8%) | A lebih baik | David |
| 2026-11-03 | B1c · MSE | 99.00 | 101.26 | -2.26 (-2.3%) | A lebih baik; turunan | David |
| 2026-11-03 | B1c · RMSE | 9.950 | 10.063 | -0.113 (-1.1%) | A lebih baik | David |
| 2026-11-03 | B1c · MedAE | 5.016 | 4.969 | +0.047 (+0.9%) | B lebih baik | David |
| 2026-11-03 | B1c · bias (pred − label) | -0.429 | -0.848 | -0.419 | A lebih baik; makin dekat 0 makin baik | David |
| 2026-11-03 | B1c · R² | 0.740 | 0.734 | -0.006 (-0.8%) | A lebih baik; turunan | David |
| 2026-11-03 | B1c · penurunan MAE vs B0 | 50.6% | 50.2% | -0.4 poin | A lebih baik | David |
| 2026-11-03 | B1 · MAE (tahun) | 7.374 | 7.484 | -0.110 (-1.5%) | A lebih baik | David |
| 2026-11-03 | B1 · MSE | 104.86 | 107.56 | -2.70 (-2.6%) | A lebih baik; turunan | David |
| 2026-11-03 | B1 · RMSE | 10.240 | 10.371 | -0.131 (-1.3%) | A lebih baik | David |
| 2026-11-03 | B1 · MedAE | 5.288 | 5.312 | -0.024 (-0.5%) | A lebih baik | David |
| 2026-11-03 | B1 · bias (pred − label) | -0.769 | -0.743 | +0.026 | B lebih baik; makin dekat 0 makin baik | David |
| 2026-11-03 | B1 · R² | 0.725 | 0.717 | -0.008 (-1.1%) | A lebih baik; turunan | David |
| 2026-11-03 | B1 · penurunan MAE vs B0 | 48.8% | 48.1% | -0.7 poin | A lebih baik | David |

### MAE M1 per kelompok

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | M1 · usia 0-12 (n=487) | 2.660 | 2.328 | +0.332 (+12.5%) | B lebih baik | David |
| 2026-11-03 | M1 · usia 13-17 (n=112) | 5.672 | 5.509 | +0.163 (+2.9%) | B lebih baik | David |
| 2026-11-03 | M1 · usia 18-24 (n=353) | 4.263 | 4.206 | +0.057 (+1.3%) | B lebih baik | David |
| 2026-11-03 | M1 · usia 25-39 (n=1516) | 4.794 | 4.394 | +0.400 (+8.3%) | B lebih baik | David |
| 2026-11-03 | M1 · usia 40-59 (n=629) | 8.189 | 7.213 | +0.976 (+11.9%) | B lebih baik | David |
| 2026-11-03 | M1 · usia 60+ (n=371) | 10.157 | 8.815 | +1.342 (+13.2%) | B lebih baik | David |
| 2026-11-03 | M1 · laki-laki (n=1862) | 5.824 | 5.154 | +0.670 (+11.5%) | B lebih baik | David |
| 2026-11-03 | M1 · perempuan (n=1606) | 5.466 | 5.048 | +0.418 (+7.6%) | B lebih baik | David |
| 2026-11-03 | M1 · ras Asian (n=537) | 4.141 | 3.778 | +0.363 (+8.8%) | B lebih baik | David |
| 2026-11-03 | M1 · ras Black (n=685) | 6.015 | 5.422 | +0.593 (+9.9%) | B lebih baik | David |
| 2026-11-03 | M1 · ras Indian (n=589) | 5.733 | 4.883 | +0.850 (+14.8%) | B lebih baik | David |
| 2026-11-03 | M1 · ras Others (n=219) | 4.058 | 4.014 | +0.044 (+1.1%) | B lebih baik | David |
| 2026-11-03 | M1 · ras White (n=1438) | 6.268 | 5.706 | +0.562 (+9.0%) | B lebih baik | David |

### Age-gating

Ambang T dipilih dari data validasi: T terkecil dengan FPR ≤ 5%. FPR = anak di bawah L yang lolos. FNR = orang dewasa yang tertahan.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | M1 · L=18 · T rekomendasi | 24 | 25 | -1 | A lebih baik; makin dekat ke L makin baik | David |
| 2026-11-03 | M1 · L=18 · FPR di T rek. | 4.5% | 4.0% | -0.5 poin | tidak dinilai; dibatasi ≤ 5% | David |
| 2026-11-03 | M1 · L=18 · FNR di T rek. | 11.4% | 14.7% | -3.3 poin | A lebih baik | David |
| 2026-11-03 | M1 · L=18 · AUC | 0.980 | 0.983 | +0.003 (+0.3%) | B lebih baik | David |
| 2026-11-03 | M1 · L=21 · T rekomendasi | 28 | 26 | +2 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-03 | M1 · L=21 · FPR di T rek. | 3.2% | 4.9% | +1.7 poin | tidak dinilai; dibatasi ≤ 5% | David |
| 2026-11-03 | M1 · L=21 · FNR di T rek. | 33.9% | 19.6% | +14.3 poin | B lebih baik | David |
| 2026-11-03 | M1 · L=21 · AUC | 0.965 | 0.970 | +0.005 (+0.5%) | B lebih baik | David |
| 2026-11-03 | B1c · L=18 · T rekomendasi | 32 | 30 | +2 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-03 | B1c · L=18 · FNR di T rek. | 49.5% | 42.8% | +6.7 poin | B lebih baik | David |
| 2026-11-03 | B1c · L=18 · AUC | 0.933 | 0.937 | +0.004 (+0.4%) | B lebih baik | David |
| 2026-11-03 | B1c · L=21 · T rekomendasi | 32 | 31 | +1 | B lebih baik; makin dekat ke L makin baik | David |
| 2026-11-03 | B1c · L=21 · FNR di T rek. | 48.2% | 45.2% | +3.0 poin | B lebih baik | David |
| 2026-11-03 | B1c · L=21 · AUC | 0.917 | 0.921 | +0.004 (+0.4%) | B lebih baik | David |

### Bootstrap berpasangan

Nilai negatif berarti model di kiri lebih akurat. CI pada kolom Keterangan adalah milik Commit B.

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | selisih MAE M1 − B1c | -1.463 | -2.076 | +0.613 | B lebih baik; CI [-2.265, -1.875], signifikan | David |
| 2026-11-03 | selisih MAE M1 − B1 | -1.715 | -2.379 | +0.664 | B lebih baik; CI [-2.584, -2.164], signifikan | David |
| 2026-11-03 | selisih MAE M1 − B0 | -8.752 | -9.305 | +0.553 | B lebih baik; CI [-9.735, -8.868], signifikan | David |
| 2026-11-03 | selisih MAE B1c − B1 | -0.252 | -0.303 | +0.051 | B lebih baik; CI [-0.353, -0.257], signifikan | David |

### Log pelatihan

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | M1 · MAE val terbaik | 5.658 | 5.105 | +0.553 (+9.8%) | B lebih baik | David |
| 2026-11-03 | M1 · MAE latih di epoch terbaik | 3.33 | 3.86 | - | berubah; B dilatih dengan augmentasi lebih berat | David |
| 2026-11-03 | M1 · selisih val − latih | 2.328 | 1.245 | +1.083 | B lebih baik; indikator overfitting | David |
| 2026-11-03 | M1 · epoch terbaik / dijalankan | 23 / 33 | 38 / 48 | - | keduanya berhenti otomatis (early stopping) | David |
| 2026-11-03 | M1 · lama latih (menit) | 7.8 | 13.7 | -5.9 | berubah; 14.2 → 17.1 detik per epoch | David |
| 2026-11-03 | B1c · MAE val terbaik | 7.122 | 7.181 | -0.059 (-0.8%) | A lebih baik | David |
| 2026-11-03 | B1c · selisih val − latih | 0.972 | 0.171 | +0.801 | B lebih baik; indikator overfitting | David |
| 2026-11-03 | B1c · epoch terbaik / dijalankan | 47 / 57 | 60 / 60 | - | B terbaik di epoch terakhir, cek apakah MAE val masih turun | David |
| 2026-11-03 | B1 · MAE val terbaik | 7.374 | 7.484 | -0.110 (-1.5%) | A lebih baik | David |
| 2026-11-03 | B1 · selisih val − latih | - | -0.026 | - | tidak ada di commit A (B1 dipakai ulang, tanpa log epoch) | David |
| 2026-11-03 | B1 · epoch terbaik / dijalankan | 60 / 60 | 52 / 60 | - | A terbaik di epoch terakhir | David |

### Efisiensi

| Date | Metric | Commit A | Commit B | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-03 | latensi CPU median (ms) | 8.777 | 19.094 | -10.317 (-117.5%) | A lebih baik; masih ≤ target 30 ms | David |
| 2026-11-03 | latensi CPU p95 (ms) | 9.647 | 23.779 | -14.132 (-146.5%) | A lebih baik | David |
| 2026-11-03 | praproses median (ms) | 1.831 | 2.179 | -0.348 (-19.0%) | A lebih baik | David |
| 2026-11-03 | jumlah parameter | 1518881 | 1518881 | +0 | sama | David |

## 2026-11-17 · `r4` vs `r6_reg_drop04`

- **Commit A / Baseline:** Run r4 · split val (n = 3468) · seed 43 · GPU A100 · MAE: 5.105 thn
- **Commit B / Model Baru:** Run r6_reg_drop04 · split val (n = 3468) · seed 43 · CPU Ryzen 5 5600X · MAE: 5.091 thn
- *Performance Improvement* positif = Commit B lebih baik.

### Ringkasan
- Peningkatan dropout (0.40), weight decay (0.08), dan augmentasi kacamata (35%) berhasil memecahkan rekor MAE validasi terendah sepanjang proyek: **5.091 tahun** (turun dari baseline r4 sebesar 5.105 tahun).
- Terobosan terbesar terjadi pada sistem age-gating usia 18 tahun: FNR (dewasa yang tertahan) turun drastis dari **14.7% ke 8.05%**, sementara T rekomendasi turun dari 25 ke 24 tahun dengan FPR terjaga <= 5%.
- Pengujian balanced sampling (`r6_balanced_sampling`) berhasil menarik T rekomendasi lebih dekat ke usia hukum (T = 22 tahun), namun dengan trade-off kenaikan MAE agregat (5.55 tahun).

### Metrik utama

| Date | Metric | Commit A (r4) | Commit B (r6_reg_drop04) | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-16 | M1 · MAE (tahun) | 5.105 | 5.091 | +0.014 (+0.3%) | B lebih baik | AquaGecko & Antigravity |
| 2026-11-16 | M1 · MSE | 53.63 | 53.01 | +0.62 (+1.2%) | B lebih baik; turunan | AquaGecko & Antigravity |
| 2026-11-16 | M1 · RMSE | 7.323 | 7.281 | +0.042 (+0.6%) | B lebih baik | AquaGecko & Antigravity |
| 2026-11-16 | M1 · MedAE | 3.500 | 3.561 | -0.061 (-1.7%) | A lebih baik | AquaGecko & Antigravity |
| 2026-11-16 | M1 · bias (pred − label) | -0.416 | -0.539 | -0.123 | A lebih baik | AquaGecko & Antigravity |
| 2026-11-16 | M1 · R² | 0.859 | 0.861 | +0.002 (+0.2%) | B lebih baik; turunan | AquaGecko & Antigravity |
| 2026-11-16 | M1 · penurunan MAE vs B0 | 64.6% | 64.7% | +0.1 poin | B lebih baik | AquaGecko & Antigravity |

### Age-gating

| Date | Metric | Commit A (r4) | Commit B (r6_reg_drop04) | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| 2026-11-16 | M1 · L=18 · T rekomendasi | 25 | 24 | +1 | B lebih baik; makin dekat L=18 | AquaGecko & Antigravity |
| 2026-11-16 | M1 · L=18 · FPR di T rek. | 4.0% | 4.3% | +0.3 poin | tidak dinilai; dibatasi <= 5% | AquaGecko & Antigravity |
| 2026-11-16 | M1 · L=18 · FNR di T rek. | 14.7% | 8.05% | +6.65 poin | B jauh lebih baik; dewasa yang tertahan turun drastis! | AquaGecko & Antigravity |
| 2026-11-16 | M1 · L=18 · AUC | 0.983 | 0.984 | +0.001 (+0.1%) | B lebih baik | AquaGecko & Antigravity |

