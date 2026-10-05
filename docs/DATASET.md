# Data UTKFace

Sumber ilmiah dan ketentuan penggunaan: [UTKFace dari pengelola](https://susanqq.github.io/UTKFace/) dan [repo AICIP](https://github.com/aicip/UTKFace).

## Unduhan alternatif

Gunakan [UTKFace Aligned & Cropped di Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped). Ini salinan dari pengunggah pihak ketiga, bukan situs pengelola asli. Buka halaman dataset, login jika diminta, lalu pilih Download. Ekstrak JPG/PNG ke `data/raw/UTKFace/` dan pertahankan nama berkas asli. Program menerima subfolder dan akhiran nama seperti `.jpg.chip.jpg`.

Pada pemeriksaan 6 Oktober 2026, halaman resmi masih terbaca, tetapi tautan Google Drive untuk aligned/cropped faces mengembalikan 404. URL GitHub lama `https://github.com/susanqq/UTKFace` juga mengembalikan 404; repo `aicip/UTKFace` masih terbaca. Halaman alternatif Kaggle dapat dibuka; unduhan arsip dan kesamaan seluruh isinya belum diverifikasi dalam pemeriksaan tautan ini.

Catat URL pengunggah/versi arsip Kaggle yang benar-benar dipakai sebagai sumber perolehan data; tetap atribusikan dataset dan paper kepada pengembang asli.

Dataset memuat lebih dari 20.000 foto dengan label usia 0-116 tahun. Situs menyatakan label usia diestimasi melalui DEX dan diperiksa oleh anotator manusia. Kesesuaian terhadap label ini tidak membuktikan usia kronologis setiap individu. Penggunaan dibatasi untuk penelitian nonkomersial; hak cipta gambar tetap pada pemilik asli.

Program melakukan audit aktual setelah arsip diperoleh. Jumlah akhir, distribusi, file rusak, konflik label, dan split dicatat dari file yang tersedia, bukan dari perkiraan ukuran katalog. Audit tidak menjamin semua wajah berasal dari individu berbeda. Kemiripan dHash membantu mencegah pasangan sangat mirip masuk split berbeda, tetapi perlu peninjauan terhadap false positive dan false negative.

Catat tanggal pengambilan arsip dan versi/sumber yang dipakai pada dokumentasi eksperimen. Jangan menambah data uji atau mengubah threshold pengelompokan setelah melihat kinerja uji. Bila manifest berubah, buat ulang seluruh eksperimen yang dibandingkan.

Foto dari demo harus merupakan foto yang penggunaannya disetujui. Tidak ada unggahan foto otomatis ke layanan lain; inferensi berjalan pada mesin yang menjalankan Streamlit.
