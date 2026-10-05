# Analisis galat

Mulai dari `evaluation_test/worst_examples.csv` setelah konfigurasi dibekukan. Buka citra asli secara lokal, tinjau contoh galat kecil sebagai pembanding, dan tulis kategori berdasarkan pengamatan. Nilai prediksi tidak membuktikan penyebab kegagalan.

| Berkas lokal | Label usia | Prediksi mentah | Galat absolut | Kondisi yang diamati | Hipotesis | Rencana pengujian |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Kategori yang dapat diperiksa antara lain pose, resolusi, pencahayaan, occlusion, crop, kelompok usia langka, dan kemungkinan masalah label. Gunakan satu perubahan yang dapat diuji dengan kontrol untuk iterasi berikutnya, lalu evaluasi melalui validasi. Hindari memilih perubahan berdasarkan test set yang sudah dibuka.

Apabila citra tidak boleh diterbitkan kembali, gunakan ID/path lokal dan deskripsi yang memadai pada catatan internal. Grafik agregat dapat dipakai untuk laporan tanpa menyalin wajah.
