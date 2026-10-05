# Catatan verifikasi teknis

Verifikasi lokal pada 6 Oktober 2026, Windows, Python 3.12 dengan lingkungan `.venv` CPU. Versi dependensi lengkap tercatat di `requirements-lock-cpu.txt`.

| Pemeriksaan | Hasil yang diamati |
| --- | --- |
| `python -m pytest -q` | 18 passed in 20.85s |
| `python -m pip check` | No broken requirements found |
| `python -m compileall -q faceage app.py scripts` | Selesai tanpa galat |
| `git diff --check` | Selesai tanpa galat pada diff yang dilacak Git |
| Smoke test 30 gambar sintetis | Audit, B0/B1/M1/B1 control satu epoch, perbandingan, dan inferensi berhasil |
| MobileNetV3 Small ImageNet resmi | Bobot berhasil diunduh dan dimuat; keluaran regresi finite dengan bentuk `(1, 1)` |
| Demo pada browser lokal | Halaman Estimasi dan Hasil eksperimen terbaca; model belum tersedia dan tombol estimasi tidak aktif |
| Logbook CSV | 13 baris isian kosong; tidak ada aktivitas, anggota, atau bukti yang dikarang |
| Notebook `Estimasi_Usia_Wajah.ipynb` | Format nbformat 4.5 valid; seluruh 13 sel kode berjalan tanpa galat dengan flag default |
| Dukungan Colab dan seluruh tes | 24 passed in 25.61s setelah penambahan 6 kasus ekstraksi ZIP |
| Notebook Colab | Format dan sintaks valid; bootstrap arsip, audit, empat metode satu epoch, evaluasi, dan grafik berjalan pada simulasi lokal dengan 30 gambar sintetis |
| Paket kode Colab | Tidak berisi dataset, model, `.venv`, `.git`, atau dokumen proposal |

Pengujian memeriksa metrik dengan contoh yang dapat dihitung manual, duplikat dan pemisahan split, path input, fingerprint manifest, checkpoint yang tertukar, pembekuan backbone/BatchNorm, pipeline CLI, serta keadaan demo tanpa model.

Luaran smoke tersimpan di `artifacts/local/smoke/smoke_summary.json`, diabaikan Git. Smoke memakai pola acak dan bobot awal acak untuk membuktikan alur teknis. Verifikasi bobot ImageNet hanya memeriksa kompatibilitas pemuatan, bukan kemampuan estimasi usia.

Dataset UTKFace belum diunduh atau dilatih penuh dalam verifikasi ini. Belum tersedia MAE/RMSE penelitian, pembuktian target proposal, evaluasi generalisasi, atau pengukuran GPU. Jalankan langkah data dan eksperimen di README untuk memperoleh hasil tersebut. Logbook tetap diisi sendiri oleh pengguna.

Notebook diserahkan tanpa output penelitian. Bukti eksekusi default berada di `artifacts/local/notebook_default_verified.ipynb` dan diabaikan Git; flag audit, pelatihan, evaluasi akhir, serta smoke dinonaktifkan pada eksekusi ini.

Bukti simulasi notebook Colab tersimpan di `artifacts/local/colab_local_simulation_verified.ipynb`. Simulasi memakai CPU lokal, folder pengganti Drive, serta konfigurasi sintetis; tidak memasang akun Google atau menguji GPU/mount Drive nyata. Versi Colab diserahkan tanpa output. Login dan eksperimen UTKFace sebenarnya perlu dijalankan pengguna sesuai `docs/COLAB.md`.
