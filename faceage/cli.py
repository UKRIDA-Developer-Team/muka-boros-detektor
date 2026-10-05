import argparse
import json
import sys
from pathlib import Path

from .data import prepare_data
from .io import read_json


def parser():
    cli=argparse.ArgumentParser(description='Estimasi usia wajah UTKFace dengan MobileNetV3 Small')
    commands=cli.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Audit dataset dan buat manifest split tetap')
    prepare.add_argument('--data',required=True,type=Path)
    prepare.add_argument('--output',type=Path,default=Path('data/processed'))
    prepare.add_argument('--seed',type=int,default=42)
    prepare.add_argument('--near-threshold',type=int,default=4)
    prepare.add_argument('--no-near',action='store_true')
    prepare.add_argument('--force',action='store_true')
    training=commands.add_parser('train',help='Latih baseline atau model utama')
    training.add_argument('--method',required=True,choices=['b0','b1','m1','b1_control'])
    training.add_argument('--data',required=True,type=Path)
    training.add_argument('--manifest',type=Path,default=Path('data/processed/manifest.csv'))
    training.add_argument('--output',required=True,type=Path)
    training.add_argument('--config',type=Path,default=Path('configs/default.json'))
    training.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    training.add_argument('--init-checkpoint',type=Path)
    for name in ['evaluate','compare']:
        command=commands.add_parser(name,help='Evaluasi atau bandingkan pada split yang sama')
        command.add_argument('--data',required=True,type=Path)
        command.add_argument('--manifest',type=Path,default=Path('data/processed/manifest.csv'))
        command.add_argument('--split',choices=['validation','test'],default='validation')
        command.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        command.add_argument('--final-test',action='store_true')
        if name=='evaluate': command.add_argument('--run',required=True,type=Path)
        else:
            command.add_argument('--runs',required=True,type=Path,nargs='+')
            command.add_argument('--output',type=Path,default=Path('runs/comparison'))
    predict=commands.add_parser('predict',help='Inferensi satu wajah yang sudah dipotong')
    predict.add_argument('--checkpoint',required=True,type=Path)
    predict.add_argument('--image',required=True,type=Path)
    predict.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu')
    predict.add_argument('--allow-synthetic',action='store_true',help='Hanya untuk uji teknis, bukan penelitian')
    return cli


def main(argv=None):
    arguments=parser().parse_args(argv)
    try:
        if arguments.command=='prepare':
            result=prepare_data(arguments.data,arguments.output,arguments.seed,
                                None if arguments.no_near else arguments.near_threshold,arguments.force)
        elif arguments.command=='train':
            from .training import train
            config=read_json(arguments.config)
            result=train(arguments.method,arguments.data,arguments.manifest,arguments.output,
                         config,arguments.device,arguments.init_checkpoint)
        elif arguments.command=='evaluate':
            from .evaluation import evaluate
            result=evaluate(arguments.run,arguments.data,arguments.manifest,arguments.split,
                            arguments.device,arguments.final_test)
        elif arguments.command=='compare':
            from .evaluation import compare_runs
            result=compare_runs(arguments.runs,arguments.data,arguments.manifest,arguments.output,
                                arguments.split,arguments.device,arguments.final_test)
        else:
            from .inference import Predictor, decode_image
            image=decode_image(arguments.image.read_bytes())
            result=Predictor(arguments.checkpoint,arguments.device,arguments.allow_synthetic).predict(image)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print(f'Gagal: {exc}',file=sys.stderr)
        return 1
