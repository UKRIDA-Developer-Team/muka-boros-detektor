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
