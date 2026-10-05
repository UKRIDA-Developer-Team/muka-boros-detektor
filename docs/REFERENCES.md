# Rujukan dan atribusi

| Sumber kredibel | Relevansi proyek |
| --- | --- |
| [UTKFace dari pengelola](https://susanqq.github.io/UTKFace/) | Unduhan, label, rentang usia, dan batas penggunaan nonkomersial. |
| [Howard dkk 2019 Searching for MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | Dasar arsitektur backbone ringan Small. Proyek mengganti keluaran klasifikasi menjadi regresi. |
| [Rothe dkk 2015 DEX](https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf) | Metode deep learning estimasi usia tampak; juga terkait proses pelabelan UTKFace. Proyek tidak mengimplementasikan ulang ensemble DEX. |
| [Zhang Song dan Qi 2017 Age Progression Regression by Conditional Adversarial Autoencoder](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Age_ProgressionRegression_by_CVPR_2017_paper.html) | Paper pengembang dataset UTKFace. Tugas generatif paper berbeda dari regresi angka usia pada proyek. |
| [Dokumentasi torchvision MobileNetV3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html) | API bobot ImageNet dan transformasi praproses yang dipakai implementasi. |

Pustaka: PyTorch, torchvision, Pillow, NumPy, Matplotlib, Streamlit, dan pytest. Periksa lisensi masing-masing saat mendistribusikan kode/model. Hak atas foto dataset tidak diberikan oleh lisensi kode pustaka.

Bantuan eksternal: OpenAI Codex membantu menyusun proposal dan implementasi repo ini, termasuk audit, split, pelatihan, evaluasi, tes, dokumentasi, dan demo. Pemilihan serta interpretasi hasil tetap menjadi tanggung jawab kelompok. Logbook tidak diisi otomatis; setiap anggota mencatat pekerjaan yang benar-benar dilakukan.
