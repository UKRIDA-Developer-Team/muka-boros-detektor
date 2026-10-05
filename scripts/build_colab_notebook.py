"""Author the standalone notebook. This builder is not needed to run it in Colab."""
import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def markdown(text):
    CELLS.append({'cell_type': 'markdown', 'metadata': {},
                  'source': textwrap.dedent(text).strip().splitlines(keepends=True)})


def code(text, tag=None):
    source = textwrap.dedent(text).strip() + '\n'
    ast.parse(source)
    CELLS.append({'cell_type': 'code', 'execution_count': None,
                  'metadata': {'tags': [tag]} if tag else {}, 'outputs': [],
                  'source': source.splitlines(keepends=True)})


def inline_module(name):
    """Keep tested functions as ordinary visible Python, with no package loader."""
    source = (ROOT / 'faceage' / f'{name}.py').read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    removed = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            removed.update(range(node.lineno - 1, node.end_lineno))
    # Globals are shared by notebook cells; evaluate is defined before training runs.
    source = ''.join(line for index, line in enumerate(lines) if index not in removed)
    code(source, f'functions-{name}')


def build():
    CELLS.clear()
    markdown('''
        # Estimasi Usia Wajah di Google Colab
        **Notebook mandiri:** seluruh kode audit, split, model, training, evaluasi,
        grafik, dan inferensi berada di sel notebook ini. Tidak perlu clone GitHub,
        mengunggah arsip source code, atau menyalin modul proyek.

        ## Cara mulai
        1. Buat folder **My Drive / ComVis** dan unggah ZIP **UTKFace Aligned & Cropped**.
           Namai arsip `dataset.zip`; pertahankan nama asli gambar di dalamnya.
        2. Buka notebook ini di Colab dan pilih **Runtime → Change runtime type → GPU**.
        3. Jalankan sel dari atas ke bawah. Default menjalankan audit dan empat metode,
           lalu membandingkan hasil validasi. Evaluasi test dan unggah foto bersifat opsional.
        4. Hasil tersimpan di **My Drive / ComVis / runs**. Notebook berisi kode,
           bukan checkpoint yang sudah dilatih; training perlu dijalankan dahulu.

        Proyek memakai UTKFace, MobileNetV3 Small ImageNet, serta baseline B0, B1,
        M1, dan B1 control. Label dataset tidak sama dengan kepastian umur kronologis.
        Skor penelitian belum diisi. Logbook dibuat kosong untuk kamu isi sendiri.
    ''')
    markdown('''
        ## A. Pasang dependensi pendukung
        PyTorch dan torchvision menggunakan instalasi bawaan runtime Colab.
        Bobot ImageNet akan diunduh otomatis saat B1 dimulai. Sel ini memasang
        pustaka pendukung langsung; tidak membaca requirements dari proyek.
    ''')
    code('''
        import os
        import subprocess
        import sys
        from pathlib import Path

        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'numpy>=1.26,<3', 'Pillow>=10,<13', 'matplotlib>=3.8,<4',
        ], check=True)
        import torch
        import torchvision
        from IPython.display import display, Image as DisplayImage

        print('Python:', sys.version.split()[0])
        print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
        print('CUDA tersedia:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        else:
            print('Training akan memakai CPU. Pilih runtime GPU jika tersedia.')
    ''', 'colab-dependencies')
    markdown('''
        ## B. Sambungkan Drive dan tentukan dataset
        Setujui dialog akses Drive. Ubah `DRIVE_PROJECT` atau `DATASET_ZIP` bila
        lokasi/nama ZIP kamu berbeda. Folder dalam ZIP boleh dipertahankan;
        pencarian gambar berjalan secara rekursif.
    ''')
    code('''
        import os
        from pathlib import Path
        from google.colab import drive
        from IPython.display import display, Image as DisplayImage

        drive.mount('/content/drive')
        DRIVE_PROJECT = Path('/content/drive/MyDrive/ComVis')
        DATASET_ZIP = DRIVE_PROJECT / 'dataset.zip'
        RUNTIME_ROOT = Path('/content/ComVis_runtime')
        if not DATASET_ZIP.is_file():
            raise FileNotFoundError(
                f'Dataset belum tersedia: {DATASET_ZIP}. '
                'Unggah ZIP UTKFace ke Drive atau sesuaikan path pada sel ini.'
            )
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('TORCH_HOME', str(RUNTIME_ROOT / '.cache' / 'torch'))
        os.environ.setdefault('MPLCONFIGDIR', str(RUNTIME_ROOT / '.cache' / 'matplotlib'))
        print('Arsip dataset:', DATASET_ZIP)
        print('Hasil eksperimen:', DRIVE_PROJECT / 'runs')
    ''', 'colab-drive')
    markdown('''
        ## C. Fungsi penyimpanan dan metrik
        JSON menyimpan konfigurasi dan hasil; CSV menyimpan manifest/prediksi.
        Semua galat dihitung dari prediksi mentah dalam satuan tahun.
    ''')
    inline_module('io')
    inline_module('metrics')
    markdown('''
        ## D. Ekstrak dataset ke disk runtime
        ZIP disalin dari Drive ke disk sementara, lalu hanya JPG/JPEG/PNG diekstrak.
        Folder ekstraksi ditandai dengan hash isi arsip untuk mencegah campuran data.
        Setelah runtime baru, jalankan sel ini lagi; dataset di Drive tetap tersimpan.
    ''')
    inline_module('archives')
    code('''
        import hashlib
        import shutil

        def archive_hash(path):
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        def stage_dataset(source, runtime_root):
            source, runtime_root = Path(source), Path(runtime_root)
            runtime_root.mkdir(parents=True, exist_ok=True)
            local_zip = runtime_root / 'dataset.zip'
            shutil.copy2(source, local_zip)
            digest = archive_hash(local_zip)
            destination = runtime_root / f'UTKFace_{digest[:12]}'
            marker = destination / '.archive_sha256'
            if marker.is_file():
                if marker.read_text(encoding='utf-8') != digest:
                    raise RuntimeError('Hash folder berbeda; gunakan folder runtime baru.')
                print('Ekstraksi sebelumnya digunakan kembali.')
            else:
                count = extract_zip(local_zip, destination, images_only=True)
                marker.write_text(digest, encoding='utf-8')
                print(f'{count:,} citra diekstrak ke disk runtime.')
            return destination, digest

        DATA, DATASET_SHA256 = stage_dataset(DATASET_ZIP, RUNTIME_ROOT)
        print('Dataset:', DATA)
        print('SHA256 arsip:', DATASET_SHA256)
    ''', 'dataset-staging')
    markdown('''
        ## E. Fungsi audit dan pembagian data
        Kode memeriksa label usia dari nama berkas, gambar rusak, hash file/piksel,
        dan kemiripan dHash. Kelompok duplikat ditempatkan dalam satu split.
        Split menargetkan 70% latih, 15% validasi, dan 15% test dengan seed 42.
        Rasio aktual bergantung ukuran kelompok. dHash bukan verifikasi identitas.
    ''')
    inline_module('data')
    markdown('''
        ## F. Model MobileNetV3 Small dan dataset PyTorch
        Citra RGB di-resize 256 pada sisi pendek, di-center crop 224×224, lalu
        dinormalisasi dengan statistik ImageNet. Augmentasi hanya untuk data latih.
        Classifier menghasilkan satu angka usia. B1/B1 control membekukan backbone
        dan BatchNorm; M1 membuka tiga modul terakhir backbone beserta head.
    ''')
    inline_module('model')
    inline_module('dataset')
    markdown('''
        ## G. Fungsi training
        Sel ini mendefinisikan training; belum menjalankan eksperimen.
        Optimizer AdamW, loss Smooth L1, checkpoint terbaik menurut MAE validasi,
        serta early stopping digunakan oleh seluruh metode jaringan.
        M1 dan B1 control dimulai dari checkpoint B1 yang sama.
        Catatan runtime mencantumkan perangkat dan versi pustaka yang benar-benar dipakai.
    ''')
    inline_module('training')
    markdown('''
        ## H. Fungsi evaluasi, grafik, dan inferensi
        MAE, RMSE, median absolute error, dan galat per kelompok usia dilaporkan.
        Prediksi, kasus terbaik/terburuk, dan grafik disimpan ke Drive.
        Test set dikunci sampai `RUN_FINAL_TEST` diaktifkan.
        Foto inferensi harus berisi satu wajah yang sudah dipotong; tidak ada deteksi otomatis.
    ''')
    inline_module('evaluation')
    inline_module('inference')
    markdown('''
        ## 1. Konfigurasi eksperimen
        Default langsung menjalankan audit, training, dan perbandingan validasi.
        Untuk uji jalan singkat pada UTKFace, ubah `CONFIG['epochs']` menjadi 2
        sebelum training dan gunakan `RUN_PREFIX` khusus, misalnya `uji02`.
        Untuk eksperimen penuh gunakan nama baru, misalnya `colab02`, dan 30 epoch.

        Folder run lama tidak ditimpa. Tahap yang telah selesai dengan konfigurasi
        dan split sama digunakan kembali. Jika tahap terputus, gunakan nama run baru
        untuk mengulang; belum ada resume optimizer/epoch. Jika mengganti ZIP,
        seed, atau threshold kemiripan, ganti `DATA_VERSION` dan `RUN_PREFIX`.
    ''')
    code('''
        DATA_VERSION = 'utkface_v1'
        RUN_PREFIX = 'colab01'
        DEVICE = 'auto'  # auto / cpu / cuda
        NEAR_THRESHOLD = 4  # 0–8; None hanya bila audit kemiripan sengaja dinonaktifkan
        CONFIG = validate_config({
            'seed': 42,
            'epochs': 30,
            'batch_size': 32,
            'patience': 5,
            'learning_rate': 0.0001,
            'backbone_learning_rate': 0.00001,
            'weight_decay': 0.0001,
            'loss_beta': 1.0,
            'unfreeze_blocks': 3,
            'augmentation': True,
            'sampling': 'natural',
            'weights': 'IMAGENET1K_V1',
            'num_workers': 0,
            'cpu_threads': 4,
            'dataset_kind': 'utkface',
        })

        RUN_PREPARE = True
        RUN_TRAINING = True
        RUN_COMPARE = True
        RUN_FINAL_TEST = False
        SELECTED_METHOD = 'm1'  # pilihan awal; putuskan dari skor VALIDASI

        PROCESSED = DRIVE_PROJECT / 'processed' / DATA_VERSION
        MANIFEST = PROCESSED / 'manifest.csv'
        RUNS = {method: DRIVE_PROJECT / 'runs' / f'{RUN_PREFIX}_{method}'
                for method in ('b0', 'b1', 'm1', 'b1_control')}
        COMPARISON = DRIVE_PROJECT / 'runs' / f'{RUN_PREFIX}_comparison'
        print(json.dumps(CONFIG, indent=2))
    ''', 'experiment-config')
    code('''
        def prepare_notebook_data():
            audit_path = PROCESSED / 'audit.json'
            if MANIFEST.is_file():
                audit = read_json(audit_path)
                if (audit.get('seed') != CONFIG['seed']
                        or audit.get('near_threshold') != NEAR_THRESHOLD
                        or audit.get('dataset_archive_sha256') != DATASET_SHA256):
                    raise ValueError(
                        'Dataset atau pengaturan audit berbeda dari manifest tersimpan. '
                        'Gunakan DATA_VERSION dan RUN_PREFIX baru.'
                    )
                saved_rows = load_manifest(MANIFEST)
                if manifest_fingerprint(saved_rows) != audit.get('fingerprint'):
                    raise ValueError('Manifest berubah dari hasil audit tersimpan.')
                verify_data_files(DATA, saved_rows)
                print('Manifest tersimpan digunakan kembali.')
                return audit
            if not RUN_PREPARE:
                raise ValueError('Manifest belum ada. Aktifkan RUN_PREPARE lalu jalankan audit.')
            print('Audit data dimulai; puluhan ribu gambar dapat memerlukan beberapa menit.', flush=True)
            audit = prepare_data(DATA, PROCESSED, seed=CONFIG['seed'], near_threshold=NEAR_THRESHOLD)
            audit['dataset_archive_sha256'] = DATASET_SHA256
            write_json(audit_path, audit)
            return audit

        def train_stage(method, init_checkpoint=None):
            if not RUN_TRAINING:
                print(f'{method.upper()}: training dinonaktifkan.')
                return None
            run = RUNS[method]
            metadata_path = run / 'run.json'
            if metadata_path.is_file():
                metadata = read_json(metadata_path)
                if metadata.get('status') != 'complete':
                    raise RuntimeError(f'{run.name} terputus. Gunakan RUN_PREFIX baru untuk mengulang.')
                if (metadata.get('method') != method or metadata.get('config') != CONFIG
                        or metadata.get('manifest_fingerprint') != manifest_fingerprint(load_manifest(MANIFEST))):
                    raise RuntimeError('Run memakai konfigurasi/split berbeda. Gunakan RUN_PREFIX baru.')
                checkpoint = run / ('b0.json' if method == 'b0' else 'best.pt')
                if not checkpoint.is_file():
                    raise RuntimeError('Checkpoint hilang. Gunakan RUN_PREFIX baru.')
                if method in {'m1', 'b1_control'}:
                    if not init_checkpoint or archive_hash(init_checkpoint) != metadata.get('source_checkpoint_sha256'):
                        raise RuntimeError('Checkpoint B1 awal berubah. Gunakan RUN_PREFIX baru.')
                # Verify the saved checkpoint, manifest, and data before reusing the result.
                evaluate(run, DATA, MANIFEST, device=DEVICE)
                print(f'{method.upper()}: hasil selesai di Drive digunakan kembali.')
                return metadata
            return train(method, DATA, MANIFEST, run, CONFIG, DEVICE, init_checkpoint)
    ''', 'workflow-functions')
    markdown('''
        ## 2. Jalankan audit dan lihat distribusi usia
        Tinjau `invalid_files.csv`, `near_duplicates.csv`, dan `duplicate_groups.csv`
        pada folder processed sebelum eksperimen penelitian. Audit tidak menjamin
        pemisahan orang yang sama antar-split karena UTKFace tidak menyediakan ID lengkap.
    ''')
    code('''
        audit = prepare_notebook_data()
        display(audit)
        rows = load_manifest(MANIFEST)
        print('Jumlah citra:', len(rows))
        print('Fingerprint:', manifest_fingerprint(rows))
        print('Split:', {name: sum(row['split'] == name for row in rows) for name in SPLITS})

        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(8, 4))
        for split in SPLITS:
            ages = [row['age'] for row in rows if row['split'] == split]
            ax.hist(ages, bins=list(range(0, 121, 5)), alpha=.45, label=split)
        ax.set(xlabel='Label usia (tahun)', ylabel='Jumlah citra', title='Distribusi label UTKFace')
        ax.legend()
        fig.tight_layout()
        fig.savefig(PROCESSED / 'age_distribution.png', dpi=170)
        display(fig)
        plt.close(fig)
    ''', 'audit-workflow')
    for method, heading, explanation in [
        ('b0', '3. Baseline B0 median usia', 'Median dihitung hanya dari label data latih; citra tidak digunakan.'),
        ('b1', '4. Baseline B1 backbone beku', 'Fitur ImageNet dipertahankan. Classifier dan head regresi dilatih; backbone dan BatchNorm beku.'),
        ('m1', '5. Metode M1 fine tuning', 'Mulai dari checkpoint B1 terbaik. Tiga modul terakhir backbone dan head dilatih.'),
        ('b1_control', '6. B1 control tambahan pelatihan head', 'Mulai dari checkpoint B1 yang sama dengan M1, lalu lanjutkan head dengan backbone beku. Batas epoch sama; early stopping dapat memberi jumlah epoch aktual berbeda.'),
    ]:
        markdown(f'## {heading}\n{explanation}\nCheckpoint dan history disimpan langsung ke Drive.')
        initial = ", init_checkpoint=RUNS['b1'] / 'best.pt'" if method in {'m1', 'b1_control'} else ''
        code(f"display(train_stage('{method}'{initial}))", f'train-{method}')
    markdown('''
        ## 7. Perbandingan validasi
        Pilih model dan konfigurasi dari hasil validasi. Target penurunan MAE 20%
        terhadap B0 merupakan sasaran proposal, bukan angka hasil yang diasumsikan.
        Tabel seluruh metode dan status kontrol fine tuning tersimpan ke Drive.
    ''')
    code('''
        if RUN_COMPARE:
            comparison = compare_runs(list(RUNS.values()), DATA, MANIFEST, COMPARISON,
                                      split='validation', device=DEVICE)
            display(comparison)
            display(DisplayImage(filename=str(COMPARISON / 'comparison_validation.png')))
        else:
            print('Perbandingan validasi dinonaktifkan.')

        for method, run in RUNS.items():
            scores = run / 'evaluation_validation' / 'metrics.json'
            if scores.is_file():
                print(method.upper())
                display(read_json(scores))
    ''', 'validation-comparison')
    markdown('''
        ## 8. Grafik dan kasus gagal model terpilih
        `SELECTED_METHOD` ditentukan pada sel konfigurasi. Grafik memakai prediksi
        mentah. Tabel kasus gagal menyimpan path dan galat; alasan kegagalan perlu
        kamu analisis dari citra, bukan dihasilkan otomatis sebagai fakta.
    ''')
    code('''
        evaluation_dir = RUNS[SELECTED_METHOD] / 'evaluation_validation'
        for filename in ('learning_curve.png', 'prediction_vs_label.png',
                         'residual_histogram.png', 'age_group_mae.png'):
            path = evaluation_dir / filename
            if path.is_file():
                print(filename)
                display(DisplayImage(filename=str(path)))
        worst = evaluation_dir / 'worst_examples.csv'
        if worst.is_file():
            display(read_csv(worst)[:10])
        else:
            print('Jalankan training dan evaluasi validasi dahulu.')
    ''', 'results-visualization')
    markdown('''
        ## 9. Evaluasi akhir test set
        Setelah model dan konfigurasi dipilih, ubah `RUN_FINAL_TEST = True` pada
        sel konfigurasi, set flag training/compare ke False, jalankan ulang konfigurasi
        dan sel ini. Jangan memilih model atau menyetel parameter berdasarkan skor test.
    ''')
    code('''
        if RUN_FINAL_TEST:
            display(compare_runs(list(RUNS.values()), DATA, MANIFEST, COMPARISON,
                                 split='test', device=DEVICE, final_test=True))
            display(DisplayImage(filename=str(COMPARISON / 'comparison_test.png')))
        else:
            print('Test set belum dibuka. RUN_FINAL_TEST = False.')
    ''', 'final-test')
    markdown('''
        ## 10. Unggah foto dan prediksi di notebook
        Setelah checkpoint penelitian tersedia, ubah `RUN_PHOTO_UPLOAD = True`.
        Pilih tepat satu JPG/PNG wajah yang sudah dipotong. Foto diproses pada runtime;
        sel tidak menyalinnya ke Drive. Foto dan output dapat ikut tersimpan bersama
        notebook jika output disimpan. Gunakan foto dengan izin.
    ''')
    code('''
        RUN_PHOTO_UPLOAD = False
        if RUN_PHOTO_UPLOAD:
            from google.colab import files
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Pilih tepat satu foto wajah JPG/PNG.')
            name, content = next(iter(uploaded.items()))
            image = decode_image(content)
            display(image)
            checkpoint = RUNS[SELECTED_METHOD] / ('b0.json' if SELECTED_METHOD == 'b0' else 'best.pt')
            predictor = Predictor(checkpoint, device=DEVICE)
            display(predictor.predict(image))
        else:
            print('Aktifkan RUN_PHOTO_UPLOAD setelah model terlatih tersedia.')
    ''', 'photo-inference')
    markdown('''
        ## 11. Logbook kosong dan lokasi hasil
        Sel berikut membuat 13 baris kosong bila file belum ada. Logbook yang sudah
        kamu isi tidak ditimpa. Catat pekerjaan nyata, tanggal, kendala, dan bukti sendiri.

        - `processed/<DATA_VERSION>/`: audit, manifest, distribusi usia.
        - `runs/<RUN_PREFIX>_<metode>/`: checkpoint, konfigurasi, metadata, history,
          CSV prediksi, metrik, dan grafik.
        - `runs/<RUN_PREFIX>_comparison/`: perbandingan validasi dan test bila diaktifkan.

        Disk runtime sementara. Saat sesi baru, jalankan setup, ekstraksi dataset,
        definisi fungsi, konfigurasi, dan audit lagi. Manifest yang cocok dan tahap
        selesai digunakan kembali; training yang terputus perlu nama run baru.
    ''')
    code('''
        logbook_path = DRIVE_PROJECT / 'logbook_kosong.csv'
        if not logbook_path.exists():
            logbook_fields = ['Nama', 'NIM', 'Kelompok', 'Minggu', 'Tanggal', 'Target',
                              'Kegiatan nyata', 'Keputusan atau hasil belajar', 'Bukti',
                              'Kendala', 'Rencana berikutnya']
            write_csv(logbook_path, [{field: '' for field in logbook_fields} for _ in range(13)],
                      logbook_fields)
        print('Logbook:', logbook_path)
        print('Audit dan manifest:', PROCESSED)
        print('Checkpoint, metrik, grafik:', DRIVE_PROJECT / 'runs')
    ''', 'blank-logbook')
    markdown('''
        ## Referensi dan batas verifikasi
        - Dataset dan ketentuan riset: [UTKFace AICIP](https://susanqq.github.io/UTKFace/).
        - Salinan arsip cropped: [UTKFace Aligned & Cropped di Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped).
        - Model: Howard dkk., [Searching for MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html).
        - Praproses/bobot: [torchvision MobileNetV3 Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
        - Lingkungan: [FAQ resmi Colab](https://research.google.com/colaboratory/faq.html).

        Alur notebook diuji secara lokal dengan ZIP berisi gambar sintetis, CPU,
        dan simulasi lokasi Drive tanpa folder repo. Simulasi tidak mengukur akurasi
        UTKFace atau membuktikan mount Drive/GPU pada akun Colab. Jalankan penelitian
        sebenarnya pada dataset UTKFace; catat versi dan perangkat aktual dari `run.json`.
    ''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.12'},
                             'colab': {'name': 'Estimasi_Usia_Wajah_Colab.ipynb'}, 'accelerator': 'GPU'},
                'cells': CELLS}
    for index, cell in enumerate(CELLS):
        cell['id'] = f'face-age-{index:02d}'
    destination = ROOT / 'Estimasi_Usia_Wajah_Colab.ipynb'
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{destination.name}: {len(CELLS)} cells, all project functions inline.')


if __name__ == '__main__':
    build()
