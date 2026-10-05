# Program estimasi usia wajah

Rancangan berasal dari proposal MobileNetV3 Small yang telah disetujui pengguna. Permintaan saat ini adalah implementasi langsung di repo yang sama, mengikuti panduan UTS dan UAS. Program untuk eksperimen akademik pada wajah tunggal yang sudah dipotong, bukan verifikasi identitas atau umur legal.

## Data dan eksperimen

- UTKFace diperoleh melalui situs pengelola; tidak disertakan dalam Git.
- Audit gambar, usia 0-116, duplikasi piksel, dan kemiripan dHash; kelompok duplikat tidak melintasi split. Kemiripan merupakan kandidat konservatif, bukan identitas orang yang terverifikasi.
- Manifest tetap 70/15/15 dengan stratifikasi kelompok usia dan seed 42. Proporsi mendekati target karena kelompok duplikat tidak dipisahkan.
- B0 median label latih; B1 MobileNetV3 Small ImageNet dengan backbone beku; M1 fine tuning tiga modul akhir dari checkpoint B1. B1 control melanjutkan head dari checkpoint yang sama untuk membandingkan perubahan fine tuning dengan anggaran epoch sama.
- RGB 224 piksel, resize sisi pendek 256, crop tengah, normalisasi ImageNet. Augmentasi latih saja.
- AdamW, Smooth L1 beta 1, batch 32, laju head 0.0001/backbone 0.00001, weight decay 0.0001, maksimal 30 epoch, early stopping 5 berdasarkan MAE validasi.
- MAE, RMSE, median galat absolut; jumlah sampel dan MAE per kelompok usia. Grafik residual, prediksi vs label, kurva latih, contoh terbaik/terburuk tanpa menyalin wajah ke laporan.

## Program dan luaran

Paket Python `faceage` menyediakan CLI prepare/train/evaluate/compare/predict. Streamlit menyediakan unggah satu wajah, pemilihan checkpoint lokal, inferensi dan latency. Tanpa checkpoint terlatih, demo tidak membuat angka prediksi palsu. Checkpoint smoke test diberi status sintetis dan tidak ditawarkan sebagai model penelitian.

Setiap run menyimpan konfigurasi, fingerprint manifest, hardware aktual, versi pustaka, log pelatihan, checkpoint terbaik, dan metrik validasi. Test set digunakan hanya melalui perintah eksplisit untuk evaluasi akhir. Pembandingan menolak run dengan split berbeda.

Dokumentasi meliputi README, sumber/atribusi termasuk bantuan Codex, pemetaan panduan, template kontribusi, checkpoint, dan logbook. Seluruh sel isian logbook kosong. Tidak ada status verifikasi dosen atau klaim hasil UTKFace yang diisi otomatis.

## Verifikasi

Tes nyata untuk label/berkas rusak, grup duplikat dan determinisme split, metrik, freeze/fine tuning, checkpoint, CLI dan demo. Smoke test memakai gambar geometris sintetis dan bobot acak untuk membuktikan alur teknis tanpa mengklaim kinerja UTKFace. Dataset wajah dan pelatihan penuh tidak tersedia sebagai hasil penelitian sebelum dijalankan pengguna.
