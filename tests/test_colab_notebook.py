"""Execute the delivered Colab notebook without making the repo importable."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


NOTEBOOK = Path(__file__).resolve().parents[1] / 'Estimasi_Usia_Wajah_Colab.ipynb'

ISOLATED_RUNNER = r'''
import ast
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
from PIL import Image

notebook = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
root = Path(sys.argv[2])
drive_root = root / 'drive' / 'ComVis'
drive_root.mkdir(parents=True)
runtime_root = root / 'runtime'
runtime_root.mkdir()
with zipfile.ZipFile(drive_root / 'dataset.zip', 'w') as archive:
    for index in range(30):
        age = [5, 15, 25, 45, 75][index % 5]
        pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='JPEG')
        archive.writestr(f'UTKFace/{age}_0_0_{index:03d}.jpg.chip.jpg', buffer.getvalue())

# Only Google authentication and the dependency installation are platform services.
# All notebook functions and workflow cells run from the copied notebook itself.
google = types.ModuleType('google')
colab = types.ModuleType('google.colab')
colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
google.colab = colab
sys.modules['google'] = google
sys.modules['google.colab'] = colab
assert 'faceage' not in sys.modules
namespace = {'__name__': '__main__'}
executed = 0
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    tags = cell.get('metadata', {}).get('tags', [])
    if 'colab-dependencies' in tags:
        continue  # Test env already contains the declared standard libraries.
    tree = ast.parse(''.join(cell['source']))
    if 'colab-drive' in tags:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                override = drive_root if 'DRIVE_PROJECT' in names else runtime_root if 'RUNTIME_ROOT' in names else None
                if override is not None:
                    node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(override))], keywords=[])
    exec(compile(ast.fix_missing_locations(tree), f'notebook_cell_{index}', 'exec'), namespace)
    executed += 1
    if 'experiment-config' in tags:
        namespace['CONFIG'] = namespace['validate_config']({
            'dataset_kind': 'synthetic', 'weights': 'none', 'epochs': 1,
            'batch_size': 8, 'augmentation': False, 'cpu_threads': 2,
        })
        namespace['DEVICE'] = 'cpu'
        namespace['NEAR_THRESHOLD'] = None

runs = namespace['RUNS']
manifest = namespace['MANIFEST']
data = namespace['DATA']
read_json = namespace['read_json']
summary = read_json(namespace['COMPARISON'] / 'comparison_validation.json')
assert len(summary['rows']) == 4
assert summary['controlled_finetuning']['valid_control']
for method, run in runs.items():
    assert read_json(run / 'run.json')['status'] == 'complete'
    assert (run / 'evaluation_validation' / 'prediction_vs_label.png').is_file()
    assert not (run / 'evaluation_test').exists(), 'Default workflow must not evaluate test data.'
    namespace['train_stage'](method, runs['b1'] / 'best.pt' if method in {'m1', 'b1_control'} else None)

try:
    namespace['evaluate'](runs['m1'], data, manifest, split='test', device='cpu')
except ValueError:
    pass
else:
    raise AssertionError('Test set was not locked.')
report = namespace['evaluate'](runs['m1'], data, manifest, split='test', device='cpu', final_test=True)
assert report['metrics']['count'] > 0
try:
    namespace['Predictor'](runs['m1'] / 'best.pt', device='cpu')
except ValueError:
    pass
else:
    raise AssertionError('Synthetic checkpoint accepted as a real age model.')
predictor = namespace['Predictor'](runs['m1'] / 'best.pt', device='cpu', allow_synthetic=True)
image_path = next(data.rglob('*.jpg'))
image = namespace['decode_image'](image_path.read_bytes())
assert np.isfinite(predictor.predict(image)['raw_age'])

# Verify restart data reuse and reject a changed dataset in the same manifest folder.
before = manifest.read_bytes()
namespace['prepare_notebook_data']()
assert before == manifest.read_bytes()
namespace['DATASET_SHA256'] = '0' * 64
try:
    namespace['prepare_notebook_data']()
except ValueError:
    pass
else:
    raise AssertionError('A different dataset reused the old split silently.')
namespace['CONFIG']['batch_size'] = 16
try:
    namespace['train_stage']('b1')
except RuntimeError:
    pass
else:
    raise AssertionError('A different config reused a completed run silently.')

logbook = namespace['read_csv'](drive_root / 'logbook_kosong.csv')
assert len(logbook) == 13 and all(not any(row.values()) for row in logbook)
assert 'faceage' not in sys.modules
print(f'STANDALONE PASS: {executed} cells, ZIP staging, audit, four methods, reuse, locked test, plots, inference, blank logbook.')
'''


def test_colab_runs_without_project_files(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or '').startswith('faceage'), 'Notebook still imports project code.'
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith('faceage') for alias in node.names)
    assert 'SOURCE_ZIP' not in code, 'Source archive is still required.'
    assert all(c.get('execution_count') is None and not c.get('outputs')
               for c in notebook['cells'] if c['cell_type'] == 'code')
    copied = tmp_path / 'standalone.ipynb'
    copied.write_text(json.dumps(notebook), encoding='utf-8')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable, '-I', '-c', ISOLATED_RUNNER, str(copied), str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, encoding='utf-8',
                            env=env, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'STANDALONE PASS:' in result.stdout
