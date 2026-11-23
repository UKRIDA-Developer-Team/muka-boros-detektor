# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Gambaran proyek

Proyek kelompok: estimasi usia dari foto wajah untuk age-gating. Ada dua bagian yang saling terhubung:

1. `AgeGating_MobileNetV3Small_UTKFace.ipynb`: pelatihan dan evaluasi. Hanya dijalankan di Google Colab dengan GPU. Semua hasil permanen (manifest, checkpoint, metrik, gambar) disimpan di Google Drive (`CFG['PROJECT_DIR']`), bukan di repo.
2. `age-gating/`: app Streamlit (proyek `uv`, Python 3.12) yang memakai checkpoint M1 hasil notebook.

Rancangan eksperimen mengikuti proposal di `docs/proposal estimasi usia wajah.pdf`. Hasil perbandingan antar run ada di `docs/comparison/comparison_results.md`.

## Perintah

Semua perintah app dijalankan dari folder `age-gating/`:

```bash
uv sync                                        # pasang dependensi (streamlit, torch, torchvision)
uv run streamlit run src/age_gating/app.py     # jalankan app di http://localhost:8501
```

Belum ada test suite atau linter. Cek cepat tanpa browser memakai `streamlit.testing.v1.AppTest`:

```bash
uv run python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('src/age_gating/app.py', default_timeout=60).run(); print(at.exception or 'OK')"
```

Notebook tidak bisa dijalankan penuh secara lokal (butuh Drive, dataset Kaggle, dan GPU). Untuk memverifikasi perubahan kode notebook, eksekusi sel kode secara berurutan pada data palsu kecil di CPU, misalnya dengan `uv run --no-project --with torch --with torchvision --with pandas --with scikit-learn --with matplotlib python ...`. Setelah sel `config`, timpa `CFG` (`PROJECT_DIR`, `LOCAL_DATA_DIR`, `MAX_EPOCHS=2`, `NUM_WORKERS=0`, `AMP=False`), lalu bandingkan hasilnya dengan versi sebelum diubah.

## Arsitektur notebook

- Setiap sel kode diawali penanda `# @cell: <nama>` (`config`, `audit`, `data`, `model`, `train_fn`, `train`, `unlock`, `metrics`, `gating`, `bootstrap`, `efficiency`, `demo`, dan lain-lain). Cari sel lewat penanda ini, bukan lewat nomor urut.
- Semua pengaturan ada di dict `CFG` di sel `config`. `RUN_TAG` menentukan nama folder run di Drive: `runs/<metode>_s<seed>_<RUN_TAG>/` berisi `best.pt`, `history.csv`, `config.json`, dan `done.json`. Run yang punya `done.json` dipakai ulang. Karena itu, setiap perubahan yang memengaruhi pelatihan wajib disertai `RUN_TAG` baru.
- Metode yang dibandingkan:
  - B0: menebak median usia latih.
  - B1: hanya head yang dilatih.
  - M1: fine-tuning seluruh backbone, mulai dari B1 terbaik. Ini model utama.
  - B1c (B1 control): mulai dari B1 terbaik, hanya head dilanjutkan dengan pengaturan yang sama seperti M1.
- Manifest (`manifest/manifest.csv`) membagi data train/val/test per grup duplikat dan per strata usia. Manifest dibangun sekali lalu dipakai ulang; SHA-256-nya dicatat di setiap run, dan cache prediksi (`results/preds/`) memakai SHA itu di nama berkasnya.
- Data uji dikunci dengan `TEST_UNLOCKED = False`. Ambang T age-gating selalu dipilih dari validasi (`recommend_threshold`). Jangan membuka data uji kecuali user memintanya.

Bagian berikut sengaja dijaga tetap stabil antar versi, karena run lama di Drive dan skill `bandingin-run` bergantung padanya: penanda `# @cell:`, nama key `CFG`, nama berkas di Drive, key `config.json`/`done.json`/checkpoint, nama kolom tabel output, dan format baris log epoch (`[M1_s43_r3] epoch N | loss ... | MAE latih ... | MAE val ... | ... s`).

Edit notebook lewat skill `edit-notebook`. Format JSON harus tetap `indent=2, ensure_ascii=False`, tanpa newline di akhir file, agar diff git bersih. Output sel jangan disentuh.

## Hubungan notebook dan app

- `age-gating/models/age_verification_model.pt` adalah salinan `runs/M1_s43_r3/best.pt` dari Drive. Isinya dict checkpoint (`model`, `median_age`, `epoch`, `val_mae`, `method`, `seed`), bukan model utuh. File ini terkena aturan `*.pt` di `.gitignore`, tetapi sengaja di-commit dengan `git add -f` agar app di Streamlit Cloud punya model. Checkpoint baru juga perlu `git add -f`.
- Streamlit Cloud hanya membaca `requirements.txt` di root repo (torch versi CPU), bukan `age-gating/pyproject.toml`. Dependensi baru wajib ditambahkan di kedua file. Cloud juga tidak memasang paket `age_gating`, jadi import antar modul app memakai nama modul langsung (`from model import ...`).
- `age-gating/src/age_gating/model.py` membangun ulang arsitektur dan praproses evaluasi dari notebook (`build_model`, resize 256, crop tengah 224, normalisasi ImageNet). Bila arsitektur atau praproses di notebook berubah, file ini harus disesuaikan. Checkpoint r3 dievaluasi tanpa TTA flip, jadi app juga tanpa TTA.
- App memutuskan dewasa dengan `AGE_LIMIT = 18`. Notebook merekomendasikan T = 24 untuk L = 18 (FPR validasi ≤ 5%).
- Model dilatih dengan foto UTKFace yang sudah dipotong ke wajah. Foto kamera atau unggahan yang tidak dipotong menghasilkan tebakan yang kurang akurat.

## Konvensi

- Bahasa proyek, dokumentasi, dan komunikasi: Bahasa Indonesia.
- Kode app di `age-gating/`: tanpa docstring dan tanpa komentar tambahan dari Claude.
- `age-gating/src/age_gating/app.py`: bagian judul dan teks (`st.title`, `st.write`) ditandai pemiliknya agar tidak diubah AI.
- Notebook: komentar singkat dan jelas, nama variabel deskriptif, tanpa em dash, panah, atau dekorasi lain di kode maupun markdown.
- Skill proyek ada di `.claude/skills/`, berupa instruksi saja tanpa script: `bandingin-run` (bandingkan output notebook dua commit) dan `edit-notebook` (edit `.ipynb` dengan aman).
