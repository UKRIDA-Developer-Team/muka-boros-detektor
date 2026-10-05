"""Ekstraksi ZIP untuk bootstrap Colab dan dataset lokal."""
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract_zip(archive, destination, images_only=False):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as stream:
        members = []
        seen = set()
        for entry in stream.infolist():
            name = entry.filename
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                    or ':' in name or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f'Path ZIP tidak aman: {name}')
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Path ZIP keluar folder tujuan: {name}')
            if entry.is_dir():
                continue
            if target in seen:
                raise ValueError(f'Path duplikat dalam ZIP: {name}')
            seen.add(target)
            if images_only and target.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            members.append((entry, target))
        if not members:
            raise ValueError('Arsip tidak berisi file yang diperlukan.')
        for entry, target in members:
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.open(entry) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    return len(members)
