import io
import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .io import read_json
from .model import make_transform, resolve_device
from .training import load_checkpoint


def decode_image(content):
    if not content or len(content)>10*1024*1024:
        raise ValueError('Unggah JPG atau PNG berukuran paling besar 10 MB.')
    with Image.open(io.BytesIO(content)) as original:
        if original.format not in {'JPEG','PNG'}: raise ValueError('Format gambar harus JPG atau PNG.')
        original.load()
        return ImageOps.exif_transpose(original).convert('RGB')


class Predictor:
    def __init__(self, checkpoint, device='cpu', allow_synthetic=False):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file(): raise ValueError('Checkpoint tidak ditemukan. Jalankan pelatihan dahulu.')
        self.device = resolve_device(device)
        self.model, self.median = None, None
        self.transform = make_transform(False)
        if checkpoint.suffix.lower()=='.json':
            payload=read_json(checkpoint)
            if payload.get('kind')!='train_median_baseline': raise ValueError('Berkas bukan baseline usia.')
            self.median=float(payload['median'])
            if not math.isfinite(self.median) or not 0<=self.median<=116: raise ValueError('Median baseline tidak valid.')
            self.metadata=payload['metadata']
            self.epoch=0
        else:
            self.model,self.metadata,payload=load_checkpoint(checkpoint,self.device)
            self.epoch=payload['epoch']
        if not self.metadata.get('research_model',False) and not allow_synthetic:
            raise ValueError('Checkpoint sintetis hanya untuk uji teknis, bukan estimasi usia wajah.')
        if self.model is not None:
            with torch.inference_mode(): self.model(torch.zeros(1,3,224,224,device=self.device))
            if self.device.type=='cuda': torch.cuda.synchronize(self.device)

    def predict(self, image):
        if not isinstance(image,Image.Image): raise ValueError('Masukan harus berupa gambar PIL.')
        start=time.perf_counter()
        if self.model is None: predicted=self.median
        else:
            tensor=self.transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(self.device)
            with torch.inference_mode(): predicted=float(self.model(tensor).flatten()[0].cpu())
            if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        if not math.isfinite(predicted): raise ValueError('Prediksi tidak finite; periksa model.')
        return {'raw_age':predicted,'display_age':round(min(116,max(0,predicted)),1),
                'latency_ms':(time.perf_counter()-start)*1000,
                'method':self.metadata['method'],'device':str(self.device),
                'research_model':self.metadata.get('research_model',False),
                'note':'Keluaran model terhadap label dataset; bukan penentuan umur resmi.'}


def available_models(runs_root):
    root=Path(runs_root)
    result=[]
    if not root.is_dir(): return result
    for metadata_path in sorted(root.glob('*/run.json')):
        try:
            metadata=read_json(metadata_path)
            if not isinstance(metadata,dict): continue
            if not metadata.get('research_model') or metadata.get('status')!='complete': continue
            if metadata.get('method') not in {'b0','b1','m1','b1_control'}: continue
            score=float(metadata['best_validation_mae'])
            if not math.isfinite(score) or score<0 or int(metadata['data_counts']['train'])<1: continue
            filename='b0.json' if metadata['method']=='b0' else 'best.pt'
            checkpoint=metadata_path.parent/filename
            if checkpoint.is_file(): result.append({'path':checkpoint,'metadata':metadata,'name':metadata_path.parent.name})
        except (ValueError,TypeError,KeyError,OSError): continue
    return result
