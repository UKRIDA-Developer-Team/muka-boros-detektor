import hashlib
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import SPLITS, load_manifest, manifest_fingerprint, verify_data_files
from .dataset import FaceDataset
from .io import write_csv, write_json
from .metrics import age_group, regression_metrics
from .model import PREPROCESS, configure_training, create_model, make_transform, resolve_device, set_training_mode

DEFAULT_CONFIG = {'seed':42,'epochs':30,'batch_size':32,'patience':5,
                  'learning_rate':0.0001,'backbone_learning_rate':0.00001,'weight_decay':0.0001,
                  'loss_beta':1.0,'unfreeze_blocks':3,'augmentation':True,'sampling':'natural',
                  'weights':'IMAGENET1K_V1','num_workers':0,'cpu_threads':4,'dataset_kind':'utkface'}


def validate_config(overrides=None):
    overrides = overrides or {}
    unknown = set(overrides)-set(DEFAULT_CONFIG)
    if unknown: raise ValueError(f'Konfigurasi tidak dikenal: {sorted(unknown)}')
    config = dict(DEFAULT_CONFIG,**overrides)
    for key in ['epochs','batch_size','patience','unfreeze_blocks','cpu_threads']:
        if type(config[key]) is not int or config[key]<1: raise ValueError(f'{key} harus integer positif.')
    if type(config['num_workers']) is not int or config['num_workers']<0: raise ValueError('num_workers tidak valid.')
    if type(config['seed']) is not int or not 0<=config['seed']<2**32: raise ValueError('seed tidak valid.')
    for key in ['learning_rate','backbone_learning_rate','loss_beta','weight_decay']:
        if not isinstance(config[key],(int,float)) or not math.isfinite(config[key]) or config[key]<0:
            raise ValueError(f'{key} harus angka finite nonnegatif.')
        if key!='weight_decay' and config[key]==0: raise ValueError(f'{key} harus positif.')
    if config['sampling'] not in {'natural','balanced'}: raise ValueError('Sampling tidak dikenal.')
    if config['weights'] not in {'IMAGENET1K_V1','none'}: raise ValueError('Bobot tidak dikenal.')
    if type(config['augmentation']) is not bool: raise ValueError('augmentation harus boolean.')
    if config['dataset_kind'] not in {'utkface','synthetic'}: raise ValueError('dataset_kind tidak dikenal.')
    if config['dataset_kind']=='utkface' and config['weights']=='none':
        raise ValueError('Eksperimen proposal menggunakan bobot ImageNet; bobot acak hanya untuk smoke sintetis.')
    return config


def seed_everything(seed, threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=True)


def _worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def environment_info(device):
    return {'python':platform.python_version(),'platform':platform.platform(),
            'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
            'numpy':str(np.__version__),'device':str(device),
            'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'gpu_memory_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
            'cuda':torch.version.cuda, 'cpu':platform.processor(),
            'cpu_threads':torch.get_num_threads()}


def predict_loader(model, loader, device):
    model.eval()
    predictions, targets, paths = [], [], []
    with torch.inference_mode():
        for images, labels, names in loader:
            output = model(images.to(device)).flatten().cpu().tolist()
            predictions.extend(output); targets.extend(labels.tolist()); paths.extend(names)
    return targets,predictions,paths


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path,map_location=device,weights_only=True)
    if not isinstance(payload,dict) or payload.get('schema_version')!=1:
        raise ValueError('Format checkpoint tidak didukung.')
    if payload.get('kind')!='mobilenet_v3_small_age_regression' or 'state_dict' not in payload:
        raise ValueError('Checkpoint bukan model estimasi usia proyek ini.')
    metadata = payload.get('metadata',{})
    if metadata.get('preprocess') != PREPROCESS or not metadata.get('manifest_fingerprint'):
        raise ValueError('Metadata praproses atau split checkpoint tidak valid.')
    if payload.get('epoch',0)<1: raise ValueError('Checkpoint belum menjalankan pelatihan.')
    model = create_model(weights='none')
    model.load_state_dict(payload['state_dict'],strict=True)
    model.to(device).eval()
    return model, metadata, payload


def train(method, root, manifest_path, output, config=None, device='auto', init_checkpoint=None):
    if method not in {'b0','b1','m1','b1_control'}: raise ValueError('Metode tidak dikenal.')
    config = validate_config(config)
    rows = load_manifest(manifest_path)
    fingerprint = manifest_fingerprint(rows)
    split_rows = {s:[r for r in rows if r['split']==s] for s in SPLITS}
    if any(not value for value in split_rows.values()): raise ValueError('Semua split harus memiliki citra.')
    verify_data_files(root,split_rows['train']+split_rows['validation'])
    seed_everything(config['seed'],config['cpu_threads'])
    device = resolve_device(device)
    median = statistics.median(row['age'] for row in split_rows['train'])
    if method in {'m1','b1_control'} and not init_checkpoint:
        raise ValueError('M1 dan B1 control memerlukan --init-checkpoint dari B1.')
    if method in {'b0','b1'} and init_checkpoint:
        raise ValueError('Checkpoint awal hanya untuk M1 atau B1 control.')
    source_hash = None
    model = None
    if init_checkpoint:
        model,source,_ = load_checkpoint(init_checkpoint,device)
        if source.get('method')!='b1' or source['manifest_fingerprint']!=fingerprint:
            raise ValueError('Checkpoint awal harus B1 dengan manifest yang sama.')
        for key in ['seed','batch_size','augmentation','sampling','weights','dataset_kind']:
            if source['config'][key]!=config[key]: raise ValueError(f'{key} berbeda dari checkpoint B1.')
        source_hash = hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest()
    elif method!='b0':
        model = create_model(config['weights'],median).to(device)
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Folder run sudah berisi hasil; pilih nama run baru.')
    output.mkdir(parents=True,exist_ok=True)
    metadata = {'schema_version':1,'method':method,'config':config,'preprocess':PREPROCESS,
                'manifest_fingerprint':fingerprint,'source_checkpoint_sha256':source_hash,
                'data_counts':{s:len(v) for s,v in split_rows.items()},
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'environment':environment_info(device),'train_median':median,
                'status':'training','research_model':config['dataset_kind']=='utkface' and config['weights']!='none'}
    write_json(output/'run.json',metadata); write_json(output/'config.json',config)
    start = time.perf_counter()
    if method=='b0':
        targets=[r['age'] for r in split_rows['validation']]
        metrics=regression_metrics(targets,[median]*len(targets))
        metadata.update({'status':'complete','training_seconds':time.perf_counter()-start,
                         'best_epoch':0,'best_validation_mae':metrics['mae'],'parameter_count':0})
        write_json(output/'b0.json',{'kind':'train_median_baseline','median':median,'metadata':metadata})
        write_csv(output/'history.csv',[{'epoch':0,'train_mae':None,'validation_mae':metrics['mae']}],
                  ['epoch','train_mae','validation_mae'])
    else:
        configure_training(model,method,config['unfreeze_blocks'])
        train_dataset=FaceDataset(root,split_rows['train'],make_transform(True,config['augmentation']))
        validation_dataset=FaceDataset(root,split_rows['validation'],make_transform(False))
        generator=torch.Generator().manual_seed(config['seed'])
        sampler=None
        if config['sampling']=='balanced':
            counts=Counter(age_group(r['age']) for r in split_rows['train'])
            weights=[1/counts[age_group(r['age'])] for r in split_rows['train']]
            sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
        train_loader=DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=sampler is None,
                                sampler=sampler,num_workers=config['num_workers'],worker_init_fn=_worker_seed,
                                generator=generator,pin_memory=device.type=='cuda')
        validation_loader=DataLoader(validation_dataset,batch_size=config['batch_size'],shuffle=False,
                                     num_workers=config['num_workers'])
        parameters=[{'params':list(model.classifier.parameters()),'lr':config['learning_rate']}]
        backbone=[p for p in model.features.parameters() if p.requires_grad]
        if backbone: parameters.append({'params':backbone,'lr':config['backbone_learning_rate']})
        optimizer=torch.optim.AdamW(parameters,weight_decay=config['weight_decay'])
        loss_fn=torch.nn.SmoothL1Loss(beta=config['loss_beta'])
        history,best,stale,best_epoch=[],float('inf'),0,0
        for epoch in range(1,config['epochs']+1):
            set_training_mode(model,method,config['unfreeze_blocks'])
            loss_total,error_total,count=0.0,0.0,0
            for images, labels, _ in train_loader:
                images,labels=images.to(device),labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                predicted=model(images).flatten()
                loss=loss_fn(predicted,labels)
                if not torch.isfinite(loss): raise ValueError('Loss tidak finite; periksa konfigurasi dan data.')
                loss.backward(); optimizer.step()
                loss_total+=float(loss.detach())*len(labels)
                error_total+=float((predicted.detach()-labels).abs().sum())
                count+=len(labels)
            targets,predictions,_=predict_loader(model,validation_loader,device)
            val=regression_metrics(targets,predictions)
            row={'epoch':epoch,'train_loss':loss_total/count,'train_mae':error_total/count,
                 'validation_mae':val['mae'],'validation_rmse':val['rmse'],
                 'elapsed_seconds':time.perf_counter()-start}
            history.append(row)
            write_csv(output/'history.csv',history,list(row))
            print(f'{method} epoch {epoch}/{config["epochs"]} train MAE={row["train_mae"]:.3f} val MAE={val["mae"]:.3f}',flush=True)
            if val['mae']<best:
                best,best_epoch,stale=val['mae'],epoch,0
                snapshot={'schema_version':1,'kind':'mobilenet_v3_small_age_regression',
                          'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                          'epoch':epoch,'validation_mae':best,'metadata':dict(metadata)}
                temporary=output/'best.pt.tmp'; torch.save(snapshot,temporary); temporary.replace(output/'best.pt')
            else: stale+=1
            if stale>=config['patience']: break
        metadata.update({'status':'complete','best_epoch':best_epoch,'best_validation_mae':best,
                         'epochs_completed':len(history),'training_seconds':time.perf_counter()-start,
                         'parameter_count':sum(p.numel() for p in model.parameters()),
                         'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    write_json(output/'run.json',metadata)
    from .evaluation import evaluate
    evaluate(output,root,manifest_path,'validation',device=str(device))
    return metadata
