"""Uji teknis dengan citra sintetis; tidak menghasilkan model usia wajah penelitian."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from faceage.data import prepare_data
from faceage.evaluation import compare_runs
from faceage.inference import Predictor
from faceage.io import read_json, write_json
from faceage.training import train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/local/smoke'))
    args=parser.parse_args()
    root=args.output
    if root.exists() and any(root.iterdir()): parser.error('Folder smoke sudah berisi data; pilih --output baru.')
    raw=root/'raw'; raw.mkdir(parents=True)
    for index in range(30):
        age=[5,15,25,45,75][index%5]
        pixels=np.random.default_rng(index).integers(0,256,(64,64,3),dtype=np.uint8)
        Image.fromarray(pixels).save(raw/f'{age}_0_0_{index:03d}.jpg')
    audit=prepare_data(raw,root/'prepared',near_threshold=None)
    manifest=root/'prepared'/'manifest.csv'
    config=read_json(Path(__file__).resolve().parents[1]/'configs'/'smoke.json')
    for method in ['b0','b1','m1','b1_control']:
        init=root/'b1'/'best.pt' if method in {'m1','b1_control'} else None
        train(method,raw,manifest,root/method,config,device='cpu',init_checkpoint=init)
    comparison=compare_runs([root/m for m in ['b0','b1','m1','b1_control']],raw,manifest,root/'comparison')
    predictor=Predictor(root/'m1'/'best.pt',allow_synthetic=True)
    with Image.open(next(raw.glob('*.jpg'))) as image: prediction=predictor.predict(image)
    summary={'status':'passed','dataset_kind':'synthetic','images':audit['valid_images'],
             'models_exercised':[r['method'] for r in comparison['rows']],
             'prediction_is_finite':bool(np.isfinite(prediction['raw_age'])),
             'note':'Ini bukti alur teknis dengan pola acak, bukan akurasi estimasi umur wajah UTKFace.'}
    write_json(root/'smoke_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
