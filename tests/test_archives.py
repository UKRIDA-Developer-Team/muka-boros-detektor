import stat
import zipfile

import pytest

from faceage.archives import extract_zip


def test_nested_images_are_extracted_without_extra_metadata(tmp_path):
    archive = tmp_path / 'dataset.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('UTKFace/25_0_0_a.jpg.chip.jpg', b'image')
        stream.writestr('README.txt', b'notice')
    destination = tmp_path / 'images'
    assert extract_zip(archive, destination, images_only=True) == 1
    assert (destination / 'UTKFace/25_0_0_a.jpg.chip.jpg').read_bytes() == b'image'
    assert not (destination / 'README.txt').exists()


@pytest.mark.parametrize('name', ['../escape.jpg', '/escape.jpg', 'C:/escape.jpg', '..\\escape.jpg'])
def test_unsafe_zip_is_rejected_before_any_file_is_written(tmp_path, name):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('valid.jpg', b'image')
        stream.writestr(name, b'bad')
    destination = tmp_path / 'images'
    with pytest.raises(ValueError):
        extract_zip(archive, destination)
    assert not (destination / 'valid.jpg').exists()


def test_symlink_member_is_rejected(tmp_path):
    archive = tmp_path / 'link.zip'
    entry = zipfile.ZipInfo('link')
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(entry, '../outside')
    with pytest.raises(ValueError):
        extract_zip(archive, tmp_path / 'images')
