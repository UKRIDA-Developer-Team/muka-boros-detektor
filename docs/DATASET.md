# Data UTKFace

Sumber: [UTKFace dari pengelola](https://susanqq.github.io/UTKFace/). Pilih aligned and cropped faces dari tautan di halaman tersebut. Simpan lokal di `data/raw/UTKFace/` dan pertahankan nama berkas.

Dataset memuat lebih dari 20.000 foto dengan label usia 0-116 tahun. Situs menyatakan label usia diestimasi melalui DEX dan diperiksa oleh anotator manusia. Kesesuaian terhadap label ini tidak membuktikan usia kronologis setiap individu. Penggunaan dibatasi untuk penelitian nonkomersial; hak cipta gambar tetap pada pemilik asli.

Program melakukan audit aktual setelah arsip diperoleh. Jumlah akhir, distribusi, file rusak, konflik label, dan split dicatat dari file yang tersedia, bukan dari perkiraan ukuran katalog. Audit tidak menjamin semua wajah berasal dari individu berbeda. Kemiripan dHash membantu mencegah pasangan sangat mirip masuk split berbeda, tetapi perlu peninjauan terhadap false positive dan false negative.

Catat tanggal pengambilan arsip dan versi/sumber yang dipakai pada dokumentasi eksperimen. Jangan menambah data uji atau mengubah threshold pengelompokan setelah melihat kinerja uji. Bila manifest berubah, buat ulang seluruh eksperimen yang dibandingkan.

Foto dari demo harus merupakan foto yang penggunaannya disetujui. Tidak ada unggahan foto otomatis ke layanan lain; inferensi berjalan pada mesin yang menjalankan Streamlit.
