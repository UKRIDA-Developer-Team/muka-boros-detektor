import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_manifest, manifest_fingerprint, verify_data_files
from .dataset import FaceDataset
from .io import read_csv, read_json, write_csv, write_json
from .metrics import age_group, age_group_metrics, regression_metrics
from .model import make_transform, resolve_device
from .training import load_checkpoint, predict_loader


def _pyplot():
    os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def create_plots(output, targets, predictions, history_path=None):
    plt=_pyplot()
    output=Path(output)
    targets,predictions=np.asarray(targets),np.asarray(predictions)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.scatter(targets,predictions,s=12,alpha=.35,color='#2563eb')
    low=float(min(targets.min(),predictions.min(),0)); high=float(max(targets.max(),predictions.max(),116))
    ax.plot([low,high],[low,high],'--',color='#475569')
    ax.set(xlabel='Label usia (tahun)',ylabel='Prediksi mentah (tahun)',title='Prediksi terhadap label usia')
    fig.tight_layout(); fig.savefig(output/'prediction_vs_label.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(predictions-targets,bins=min(30,max(3,len(targets))),color='#2563eb',edgecolor='white')
    ax.axvline(0,color='#475569',linestyle='--')
    ax.set(xlabel='Residual prediksi dikurangi label (tahun)',ylabel='Jumlah citra',title='Distribusi residual')
    fig.tight_layout(); fig.savefig(output/'residual_histogram.png',dpi=170); plt.close(fig)
    groups=age_group_metrics(targets.tolist(),predictions.tolist())
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(list(groups),[v['mae'] if v['mae'] is not None else 0 for v in groups.values()],color='#2563eb')
    ax.set(xlabel='Kelompok label usia',ylabel='MAE (tahun)',title='Galat per kelompok usia')
    for i,v in enumerate(groups.values()): ax.text(i,0,f'n={v["count"]}',ha='center',va='bottom',fontsize=8)
    fig.tight_layout(); fig.savefig(output/'age_group_mae.png',dpi=170); plt.close(fig)
    if history_path and Path(history_path).is_file():
        rows=read_csv(history_path)
        if rows and rows[0].get('train_mae') not in {'',None}:
            fig,ax=plt.subplots(figsize=(6,4))
            ax.plot([int(r['epoch']) for r in rows],[float(r['train_mae']) for r in rows],label='Latih')
            ax.plot([int(r['epoch']) for r in rows],[float(r['validation_mae']) for r in rows],label='Validasi')
            ax.set(xlabel='Epoch',ylabel='MAE (tahun)',title='Kurva pelatihan'); ax.legend()
            fig.tight_layout(); fig.savefig(output/'learning_curve.png',dpi=170); plt.close(fig)


def evaluate(run, root, manifest_path, split='validation', device='auto', final_test=False):
    if split not in {'validation','test'}: raise ValueError('Evaluasi hanya validation atau test.')
    if split=='test' and not final_test:
        raise ValueError('Test set dikunci untuk evaluasi akhir; tambahkan --final-test setelah memilih konfigurasi.')
    run=Path(run)
    metadata=read_json(run/'run.json')
    if metadata.get('status')!='complete': raise ValueError('Run belum selesai.')
    rows=load_manifest(manifest_path)
    fingerprint=manifest_fingerprint(rows)
    if fingerprint!=metadata['manifest_fingerprint']:
        raise ValueError('Manifest berbeda dari pelatihan; perbandingan tidak valid.')
    selected=[r for r in rows if r['split']==split]
    if not selected: raise ValueError('Split evaluasi kosong.')
    verify_data_files(root,selected)
    device=resolve_device(device)
    torch.set_num_threads(metadata['config']['cpu_threads'])
    start=time.perf_counter()
    if metadata['method']=='b0':
        payload=read_json(run/'b0.json')
        targets=[r['age'] for r in selected]; predictions=[payload['median']]*len(selected)
        paths=[r['path'] for r in selected]
    else:
        model,checkpoint_metadata,_=load_checkpoint(run/'best.pt',device)
        if checkpoint_metadata['manifest_fingerprint']!=fingerprint: raise ValueError('Checkpoint dan manifest berbeda.')
        if checkpoint_metadata.get('method')!=metadata['method'] or checkpoint_metadata.get('config')!=metadata['config']:
            raise ValueError('Checkpoint tertukar atau konfigurasi run berbeda dari model yang dimuat.')
        loader=DataLoader(FaceDataset(root,selected,make_transform(False)),
                          batch_size=metadata['config']['batch_size'],shuffle=False,
                          num_workers=metadata['config']['num_workers'])
        targets,predictions,paths=predict_loader(model,loader,device)
    elapsed=time.perf_counter()-start
    scores=regression_metrics(targets,predictions)
    output=run/f'evaluation_{split}'; output.mkdir(exist_ok=True)
    prediction_rows=[{'path':path,'age':age,'age_group':age_group(age),'prediction':pred,
                      'residual':pred-age,'absolute_error':abs(pred-age)}
                     for path,age,pred in zip(paths,targets,predictions)]
    write_csv(output/'predictions.csv',prediction_rows,list(prediction_rows[0]))
    ordered=sorted(prediction_rows,key=lambda r:r['absolute_error'])
    write_csv(output/'best_examples.csv',ordered[:10],list(prediction_rows[0]))
    write_csv(output/'worst_examples.csv',list(reversed(ordered[-10:])),list(prediction_rows[0]))
    per_group=age_group_metrics(targets,predictions)
    write_csv(output/'age_groups.csv',[dict(age_group=group,**value) for group,value in per_group.items()],
              ['age_group','count','mae','rmse','median_absolute_error'])
    report={'method':metadata['method'],'split':split,'manifest_fingerprint':fingerprint,
            'dataset_kind':metadata['config']['dataset_kind'],'research_model':metadata['research_model'],
            'metrics':scores,'age_groups':per_group,'evaluation_seconds':elapsed,
            'amortized_ms_per_image_including_loading':elapsed*1000/len(targets),
            'device':str(device),'raw_predictions':True,'created_from_checkpoint':'best.pt' if metadata['method']!='b0' else 'b0.json'}
    write_json(output/'metrics.json',report)
    create_plots(output,targets,predictions,run/'history.csv')
    return report


def compare_runs(runs, root, manifest_path, output, split='validation', device='auto', final_test=False):
    runs=[Path(run) for run in runs]
    if len(runs)<2: raise ValueError('Perbandingan memerlukan minimal dua run.')
    metadata=[read_json(run/'run.json') for run in runs]
    fingerprint=manifest_fingerprint(load_manifest(manifest_path))
    if any(value.get('manifest_fingerprint')!=fingerprint for value in metadata):
        raise ValueError('Semua run harus menggunakan manifest yang sama.')
    if len({value['config']['dataset_kind'] for value in metadata})!=1:
        raise ValueError('Run sintetis dan penelitian tidak boleh dibandingkan.')
    if len(set(str(run.resolve()) for run in runs))!=len(runs): raise ValueError('Run yang sama dicantumkan dua kali.')
    rows=[]
    for run,meta in zip(runs,metadata):
        report=evaluate(run,root,manifest_path,split,device,final_test)
        rows.append({'run':run.name,'method':meta['method'],**report['metrics'],
                     'best_epoch':meta['best_epoch'],'training_seconds':meta['training_seconds'],
                     'parameter_count':meta['parameter_count']})
    baseline=next((r for r in rows if r['method']=='b0'),None)
    main=next((r for r in rows if r['method']=='m1'),None)
    improvement=None
    if baseline and main and baseline['mae']>0: improvement=(baseline['mae']-main['mae'])/baseline['mae']*100
    m1=next((m for m in metadata if m['method']=='m1'),None)
    control=next((m for m in metadata if m['method']=='b1_control'),None)
    same_source=bool(m1 and control and m1['source_checkpoint_sha256'] and
                     m1['source_checkpoint_sha256']==control['source_checkpoint_sha256'])
    controlled_keys=['epochs','batch_size','patience','learning_rate','backbone_learning_rate',
                     'weight_decay','loss_beta','seed','augmentation','sampling','dataset_kind','weights']
    same_settings=bool(m1 and control and all(m1['config'][key]==control['config'][key] for key in controlled_keys))
    summary={'split':split,'manifest_fingerprint':fingerprint,'rows':rows,
             'dataset_kind':metadata[0]['config']['dataset_kind'],
             'm1_mae_improvement_percent_over_b0':improvement,
             'target_20_percent_met':improvement>=20 if improvement is not None else None,
             'controlled_finetuning':{'same_initial_checkpoint':same_source,'same_training_settings':same_settings,
                                      'valid_control':same_source and same_settings,
                                      'note':'B1 control dan M1 memiliki batas epoch sama; early stopping dapat memberi jumlah epoch aktual berbeda.'}}
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/f'comparison_{split}.csv',rows,list(rows[0]))
    write_json(output/f'comparison_{split}.json',summary)
    plt=_pyplot(); fig,ax=plt.subplots(figsize=(7,4))
    ax.bar([r['run'] for r in rows],[r['mae'] for r in rows],color='#2563eb')
    ax.set(ylabel='MAE (tahun)',title=f'Perbandingan pada split {split}'); ax.tick_params(axis='x',labelrotation=15)
    fig.tight_layout(); fig.savefig(output/f'comparison_{split}.png',dpi=170); plt.close(fig)
    return summary
