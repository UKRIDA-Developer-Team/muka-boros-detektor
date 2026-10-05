# Pemetaan terhadap panduan proyek kuliah

Sumber persyaratan: `Panduan_Proyek_Akhir_Visi_Komputer_UTS_UAS.pdf` yang diberikan pengguna. Isi panduan digunakan sebagai persyaratan proyek; daftar mahasiswa dan instruksi pengumpulan di dalamnya tidak dianggap sebagai otorisasi mengisi kontribusi, mengunggah berkas, atau menghubungi orang lain.

| Unsur yang diperiksa | Artefak program atau dokumentasi | Bukti yang perlu dibuat saat eksperimen |
| --- | --- | --- |
| Data dan lisensi | `docs/DATASET.md`, CLI `prepare` | Arsip diperoleh, audit aktual dan contoh data diperiksa |
| Split, seed, leakage | Manifest dan fingerprint, grouped split | `manifest.csv`, `audit.json`, peninjauan pasangan mirip |
| Baseline dan metode utama | B0, B1, M1 | Log, checkpoint, dan metrik aktual setiap run |
| Eksperimen terkontrol | B1 control dan M1 dari checkpoint sama | Perbandingan pada split sama dengan konfigurasi yang tercatat |
| Metrik dan visual | CLI `evaluate` dan `compare` | CSV/JSON/grafik, kasus berhasil dan gagal, interpretasi kelompok |
| Inferensi input baru | CLI `predict`, Streamlit | Foto dengan izin, latency aktual, video demo cadangan 2-3 menit |
| Reproduksi dan dependensi | README, requirements, konfigurasi, tes | Versi lingkungan aktual, perintah dan hasil eksperimen |
| Lima rujukan termasuk paper | `docs/REFERENCES.md` | Pemahaman rujukan dan pemutakhiran daftar pustaka laporan |
| Kontribusi dan logbook | Template CSV/Markdown kosong | Diisi sendiri setiap anggota dengan pekerjaan dan bukti nyata |
| Checkpoint | Template CP00-CP13 | Tanggal, bukti, umpan balik; verifikasi hanya dosen/asisten |

## Persiapan UTS

UTS adalah sidang proposal. Program ini mendukung bukti data terbaca dan inferensi/pipeline berjalan. Smoke sintetis hanya bukti alur teknis; untuk bukti kelayakan data, unduh UTKFace dan jalankan audit pada data nyata. Siapkan proposal, slide 6-8 inti, peran dan pembagian presentasi, serta tracker sesuai panduan.

Proposal paper dua kolom memuat isi proposal lengkap, tujuh rujukan, rencana eksperimen Colab dan B1 control, serta risiko dan mitigasi. Jumlah halaman mengikuti isi dengan ukuran huruf yang mudah dibaca. Panduan menyarankan 5-7 halaman di luar referensi serta meminta pembagian peran. Nama/NIM dan kontribusi anggota masih perlu dilengkapi; versi paper ini tidak diklaim sudah memenuhi semua butir administrasi baru.

## Persiapan UAS

Pilih konfigurasi menggunakan validasi, bekukan keputusan, kemudian evaluasi test set. Tulis hasil positif maupun negatif, perbedaan tiap kelompok usia, kasus gagal, dan keterbatasan label. Setiap gambar/tabel yang dipakai pada paper harus dirujuk dalam teks.

Laporan teknis akhir, slide hasil, video cadangan, pengisian logbook, kontribusi anggota, dan unggahan Google Drive/Sheets memerlukan hasil eksperimen serta informasi kelompok. Repo menyediakan program dan format pencatatan, bukan status kelulusan atau bukti dosen sudah memverifikasi checkpoint.
