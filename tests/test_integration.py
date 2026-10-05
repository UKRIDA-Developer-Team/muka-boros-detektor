import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from faceage.data import prepare_data, load_manifest


def make_dataset(root):
    root.mkdir()
    for index in range(30):
        values=np.random.default_rng(index).integers(0,256,(64,64,3),dtype=np.uint8)
        age=[5,15,25,45,75][index%5]
        Image.fromarray(values).save(root/f'{age}_0_0_{index:03d}.jpg')


class IntegrationTests(unittest.TestCase):
    def test_real_train_evaluate_and_inference_roundtrip(self):
        self.assertIsNotNone(importlib.util.find_spec('faceage.evaluation'),'Evaluasi belum dibuat')
        self.assertIsNotNone(importlib.util.find_spec('faceage.inference'),'Inferensi belum dibuat')
        from faceage.training import train
        from faceage.evaluation import evaluate, compare_runs
        from faceage.inference import Predictor
        from faceage.io import read_json, write_csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'raw'; prepared=root/'prepared'
            make_dataset(raw)
            prepare_data(raw,prepared,near_threshold=None)
            manifest=prepared/'manifest.csv'
            config={'dataset_kind':'synthetic','weights':'none','epochs':1,'batch_size':8,
                    'augmentation':False,'cpu_threads':2}
            train('b0',raw,manifest,root/'b0',config,device='cpu')
            rows=load_manifest(manifest)
            expected=float(np.median([r['age'] for r in rows if r['split']=='train']))
            self.assertEqual(read_json(root/'b0'/'b0.json')['median'],expected)
            train('b1',raw,manifest,root/'b1',config,device='cpu')
            train('m1',raw,manifest,root/'m1',config,device='cpu',init_checkpoint=root/'b1'/'best.pt')
            train('b1_control',raw,manifest,root/'control',config,device='cpu',init_checkpoint=root/'b1'/'best.pt')
            report=evaluate(root/'m1',raw,manifest,split='test',final_test=True)
            self.assertEqual(report['metrics']['count'],sum(r['split']=='test' for r in rows))
            self.assertTrue((root/'m1'/'evaluation_test'/'predictions.csv').is_file())
            self.assertTrue((root/'m1'/'evaluation_test'/'prediction_vs_label.png').is_file())
            with self.assertRaises(ValueError): Predictor(root/'b1'/'best.pt')
            predictor=Predictor(root/'b1'/'best.pt',allow_synthetic=True)
            with Image.open(raw/rows[0]['path']) as image: result=predictor.predict(image)
            self.assertTrue(np.isfinite(result['raw_age']))
            self.assertGreaterEqual(result['latency_ms'],0)
            self.assertEqual(result['research_model'],False)
            summary=compare_runs([root/'b0',root/'b1',root/'m1',root/'control'],raw,manifest,root/'comparison')
            self.assertEqual(len(summary['rows']),4)
            self.assertTrue(summary['controlled_finetuning']['same_initial_checkpoint'])
            original=(root/'b1'/'best.pt').read_bytes()
            (root/'b1'/'best.pt').write_bytes((root/'m1'/'best.pt').read_bytes())
            with self.assertRaises(ValueError): evaluate(root/'b1',raw,manifest)
            (root/'b1'/'best.pt').write_bytes(original)
            changed=[dict(r) for r in rows]; changed[0]['age']+=1
            altered=root/'different.csv'; write_csv(altered,changed,list(changed[0]))
            with self.assertRaises(ValueError): evaluate(root/'b1',raw,altered)
            with self.assertRaises(ValueError): train('b0',raw,manifest,root/'b0',config)
            with self.assertRaises(ValueError): evaluate(root/'b1',raw,manifest,'test')

    def test_cli_baseline_and_predict_output(self):
        self.assertIsNotNone(importlib.util.find_spec('faceage.cli'),'CLI belum dibuat')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'raw'; make_dataset(raw)
            def run(*arguments):
                return subprocess.run([sys.executable,'-m','faceage',*map(str,arguments)],
                                      capture_output=True,text=True,encoding='utf-8',timeout=90)
            result=run('prepare','--data',raw,'--output',root/'prepared','--no-near')
            self.assertEqual(result.returncode,0,result.stderr)
            result=run('train','--method','b0','--data',raw,'--manifest',root/'prepared'/'manifest.csv','--output',root/'b0')
            self.assertEqual(result.returncode,0,result.stderr)
            image=next(raw.glob('*.jpg'))
            result=run('predict','--checkpoint',root/'b0'/'b0.json','--image',image)
            self.assertEqual(result.returncode,0,result.stderr)
            payload=json.loads(result.stdout)
            self.assertIn('raw_age',payload)
            result=run('predict','--checkpoint',root/'missing.pt','--image',image)
            self.assertNotEqual(result.returncode,0)

    def test_prepare_will_not_silently_replace_existing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'raw'; make_dataset(raw)
            prepare_data(raw,root/'prepared',near_threshold=None)
            before=(root/'prepared'/'manifest.csv').read_bytes()
            with self.assertRaises(ValueError): prepare_data(raw,root/'prepared',near_threshold=None)
            self.assertEqual((root/'prepared'/'manifest.csv').read_bytes(),before)
