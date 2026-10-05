"""Buat paket kode Colab tanpa dataset, checkpoint, atau lingkungan lokal."""
import argparse
import zipfile
from pathlib import Path


def package(output):
    root = Path(__file__).resolve().parents[1]
    files = []
    for folder, suffixes in [('faceage', {'.py'}), ('configs', {'.json'}),
                             ('scripts', {'.py', '.ps1'}), ('tests', {'.py'}),
                             ('docs', {'.md', '.csv'})]:
        files.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and path.suffix in suffixes)
    for name in ['app.py', 'README.md', 'pytest.ini', '.gitignore',
                 'requirements.txt', 'requirements-colab.txt', 'requirements-notebook.txt',
                 'Estimasi_Usia_Wajah.ipynb', 'Estimasi_Usia_Wajah_Colab.ipynb',
                 '.streamlit/config.toml']:
        path = root / name
        if path.is_file():
            files.append(path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root).as_posix())
    print(f'{output.resolve()}: {len(set(files))} files, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/local/ComVis_Colab_Source.zip'))
    package(parser.parse_args().output)
