import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import numpy as np


def load_data_module(test):
    test.assertIsNotNone(importlib.util.find_spec('faceage'), 'Data pipeline belum dibuat')
    return importlib.import_module('faceage.data')


def sample_image(path, seed):
    values = np.random.default_rng(seed).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(values).save(path, format='PNG')


class DataTests(unittest.TestCase):
    def test_age_labels_reject_invalid_range_and_malformed_names(self):
        data = load_data_module(self)
        self.assertEqual(data.parse_age('27_0_2_20170101.jpg'), 27)
        self.assertEqual(data.parse_age('0_1_0_photo.jpg'), 0)
        self.assertEqual(data.parse_age('116_0_0_photo.jpg'), 116)
        for name in ['117_0_0_photo.jpg', '-1_0_0_photo.jpg', 'age_1.jpg', '27.jpg']:
            with self.assertRaises(ValueError):
                data.parse_age(name)

    def test_audit_rejects_corrupt_images_and_groups_renamed_pixels(self):
        data = load_data_module(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_image(root/'10_0_0_a.jpg', 1)
            sample_image(root/'50_1_1_b.jpg', 1)
            sample_image(root/'90_0_2_c.jpg', 2)
            (root/'20_0_0_bad.jpg').write_bytes(b'not an image')
            sample_image(root/'117_0_0_badage.jpg', 3)
            rows, report, invalid, near = data.audit_images(root, near_threshold=None)
            self.assertEqual(report['valid_images'], 3)
            self.assertEqual(len(invalid), 2)
            self.assertEqual(rows[0]['group_id'], rows[1]['group_id'])
            self.assertNotEqual(rows[0]['group_id'], rows[2]['group_id'])
            self.assertEqual(report['conflicting_age_groups'], 1)

    def test_group_split_is_deterministic_and_never_leaks_duplicates(self):
        data = load_data_module(self)
        rows = []
        for group in range(40):
            for copy in range(1 + group % 3):
                rows.append({'path':f'{group}_{copy}.jpg', 'age': 30,
                             'group_id':f'g{group:03d}', 'pixel_sha256':f'h{group}'})
        one = data.assign_splits(rows, seed=42)
        two = data.assign_splits(list(reversed(rows)), seed=42)
        self.assertEqual({r['path']:r['split'] for r in one}, {r['path']:r['split'] for r in two})
        groups = {}
        for row in one:
            groups.setdefault(row['group_id'], set()).add(row['split'])
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        self.assertEqual(set(r['split'] for r in one), {'train','validation','test'})
        counts = {s:sum(r['split']==s for r in one) for s in ['train','validation','test']}
        self.assertLess(abs(counts['train']/len(one)-0.7), 0.08)

    def test_manifest_validation_rejects_path_escape_and_duplicate_leakage(self):
        data = load_data_module(self)
        base = [{'path':'a.jpg','age':20,'split':'train','group_id':'same','pixel_sha256':'h'},
                {'path':'b.jpg','age':20,'split':'test','group_id':'same','pixel_sha256':'h'}]
        with self.assertRaises(ValueError): data.validate_manifest(base)
        with self.assertRaises(ValueError):
            data.validate_manifest([dict(base[0],path='../outside.jpg')])

    def test_less_than_three_groups_cannot_create_research_split(self):
        data = load_data_module(self)
        with self.assertRaises(ValueError):
            data.assign_splits([{'path':'a.jpg','age':20,'group_id':'one'}])
