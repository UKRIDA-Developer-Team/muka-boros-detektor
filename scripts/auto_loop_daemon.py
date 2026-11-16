#!/usr/bin/env python3
"""
scripts/auto_loop_daemon.py
Pengontrol pelatihan berkelanjutan di latar belakang (autonomous overnight training loop).
- Mengeksekusi serangkaian eksperimen penalaan hyperparameter dan regularisasi.
- Memantau MAE validasi dan mencegah overfitting.
- Otomatis membuat commit git saat metrik model meningkat melampaui performa terbaik sebelumnya.
- Berjalan terus-menerus hingga file 'stop_training.flag' dibuat atau proses dihentikan.
"""

import os
import sys
import time
import json
import subprocess
import datetime
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / 'runs' / 'overnight_training.log'
STATUS_FILE = ROOT / 'runs' / 'current_status.json'
BEST_METRICS_FILE = ROOT / 'runs' / 'best_metrics.json'
STOP_FLAG = ROOT / 'stop_training.flag'
COMPARISON_FILE = ROOT / 'docs' / 'comparison' / 'comparison_results.md'
VENV_PYTHON = ROOT / '.venv' / 'bin' / 'python'

def log(msg):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def update_status(data):
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_best_metrics():
    if BEST_METRICS_FILE.exists():
        with open(BEST_METRICS_FILE, 'r') as f:
            return json.load(f)
    # Default baseline dari Commit B (M1 r4) di docs/comparison/comparison_results.md
    return {
        'MAE': 5.105,
        'RMSE': 7.323,
        'MedAE': 3.500,
        'bias': -0.416,
        'R2': 0.859,
        'run_tag': 'baseline_r4',
        'updated_at': '2026-11-03'
    }

def save_best_metrics(metrics):
    with open(BEST_METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

def append_comparison_record(run_tag, old_metrics, new_metrics):
    if not COMPARISON_FILE.exists():
        return
    date_str = datetime.date.today().isoformat()
    mae_diff = old_metrics['MAE'] - new_metrics['MAE']
    pct_imp = (mae_diff / old_metrics['MAE']) * 100.0

    entry = f"""
## {date_str} · `{old_metrics['run_tag']}` vs `{run_tag}` (Autonomous Run)

- **Baseline:** Run `{old_metrics['run_tag']}` · MAE: {old_metrics['MAE']} thn
- **Model Baru:** Run `{run_tag}` · MAE: {new_metrics['MAE']} thn · Improvisasi: {pct_imp:+.2f}%

| Date | Metric | Baseline ({old_metrics['run_tag']}) | Baru ({run_tag}) | Performance Improvement | Keterangan | Author |
|---|---|---|---|---|---|---|
| {date_str} | M1 · MAE (tahun) | {old_metrics['MAE']:.3f} | {new_metrics['MAE']:.3f} | {mae_diff:+.3f} ({pct_imp:+.1f}%) | {'B lebih baik' if mae_diff > 0 else 'A lebih baik'} | Auto-Tuner |
| {date_str} | M1 · RMSE | {old_metrics.get('RMSE', '-')} | {new_metrics.get('RMSE', '-')} | - | turunan | Auto-Tuner |
| {date_str} | M1 · MedAE | {old_metrics.get('MedAE', '-')} | {new_metrics.get('MedAE', '-')} | - | {'B lebih baik' if new_metrics.get('MedAE', 99) < old_metrics.get('MedAE', 99) else 'A lebih baik'} | Auto-Tuner |
| {date_str} | M1 · bias | {old_metrics.get('bias', '-')} | {new_metrics.get('bias', '-')} | - | makin dekat 0 makin baik | Auto-Tuner |
"""
    with open(COMPARISON_FILE, 'a', encoding='utf-8') as f:
        f.write(entry + '\n')

def git_commit_improvement(run_tag, new_metrics, mae_diff):
    try:
        subprocess.run(['git', 'add', 'runs/', 'results/', 'docs/comparison/'], cwd=str(ROOT), check=True)
        commit_msg = (f"model: capai MAE {new_metrics['MAE']:.3f} thn ({mae_diff:+.3f} thn vs baseline) "
                      f"pada run {run_tag}")
        res = subprocess.run(['git', 'commit', '-m', commit_msg], cwd=str(ROOT), capture_output=True, text=True)
        log(f"Git commit berhasil: {commit_msg}")
        log(f"Git output: {res.stdout.strip()}")
    except Exception as e:
        log(f"Gagal melakukan git commit: {e}")

def generate_experiments():
    """Daftar konfigurasi eksperimen yang dieksplorasi berurutan."""
    configs = [
        # Eksperimen 1: Regularisasi & Dropout lebih tinggi untuk cegah overfitting usia tua
        {
            'run_tag': 'r6_reg_drop04',
            'epochs': 35,
            'patience': 7,
            'lr_head': 3e-4,
            'lr_backbone': 9e-5,
            'weight_decay': 0.08,
            'dropout': 0.40,
            'smooth_l1_beta': 1.0,
            'aug_glasses_p': 0.35,
            'aug_rotate': 12,
            'aug_translate': 0.06,
            'balanced_sampling': False
        },
        # Eksperimen 2: Balanced Age Sampling (meratakan porsi usia remaja dan lanjut usia)
        {
            'run_tag': 'r6_balanced_sampling',
            'epochs': 35,
            'patience': 7,
            'lr_head': 3e-4,
            'lr_backbone': 8e-5,
            'weight_decay': 0.06,
            'dropout': 0.35,
            'smooth_l1_beta': 1.0,
            'aug_glasses_p': 0.30,
            'aug_rotate': 10,
            'aug_translate': 0.05,
            'balanced_sampling': True
        },
        # Eksperimen 3: Milder Loss Beta untuk penalti galat kecil lebih tajam
        {
            'run_tag': 'r6_beta08_wd07',
            'epochs': 35,
            'patience': 7,
            'lr_head': 2.5e-4,
            'lr_backbone': 7.5e-5,
            'weight_decay': 0.07,
            'dropout': 0.35,
            'smooth_l1_beta': 0.8,
            'aug_glasses_p': 0.35,
            'aug_rotate': 12,
            'aug_translate': 0.06,
            'balanced_sampling': False
        },
        # Eksperimen 4: Conservative Backbone LR dengan augmentasi rotasi lebih luas
        {
            'run_tag': 'r6_rot15_lrbackbone6e5',
            'epochs': 40,
            'patience': 8,
            'lr_head': 3e-4,
            'lr_backbone': 6e-5,
            'weight_decay': 0.05,
            'dropout': 0.30,
            'smooth_l1_beta': 1.0,
            'aug_glasses_p': 0.30,
            'aug_rotate': 15,
            'aug_translate': 0.07,
            'balanced_sampling': False
        },
        # Eksperimen 5: Kombinasi terbaik: Balanced sampling + dropout 0.4 + beta 0.9
        {
            'run_tag': 'r6_combo_balanced_drop04',
            'epochs': 40,
            'patience': 8,
            'lr_head': 2.5e-4,
            'lr_backbone': 7e-5,
            'weight_decay': 0.08,
            'dropout': 0.40,
            'smooth_l1_beta': 0.9,
            'aug_glasses_p': 0.35,
            'aug_rotate': 12,
            'aug_translate': 0.06,
            'balanced_sampling': True
        }
    ]
    return configs

def run_experiment(exp_cfg):
    run_tag = exp_cfg['run_tag']
    log(f"==================================================")
    log(f"Mulai eksperimen: {run_tag}")
    log(f"Parameter: {json.dumps(exp_cfg)}")

    cmd = [
        str(VENV_PYTHON),
        str(ROOT / 'scripts' / 'train_runner.py'),
        '--run-tag', run_tag,
        '--epochs', str(exp_cfg['epochs']),
        '--patience', str(exp_cfg['patience']),
        '--lr-head', str(exp_cfg['lr_head']),
        '--lr-backbone', str(exp_cfg['lr_backbone']),
        '--weight-decay', str(exp_cfg['weight_decay']),
        '--dropout', str(exp_cfg['dropout']),
        '--smooth-l1-beta', str(exp_cfg['smooth_l1_beta']),
        '--aug-glasses-p', str(exp_cfg['aug_glasses_p']),
        '--aug-rotate', str(exp_cfg['aug_rotate']),
        '--aug-translate', str(exp_cfg['aug_translate']),
        '--batch-size', '64',
        '--num-workers', '4'
    ]
    if exp_cfg.get('balanced_sampling', False):
        cmd.append('--balanced-sampling')

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
    
    last_log_line = ""
    for line in iter(proc.stdout.readline, ''):
        line = line.strip()
        if not line:
            continue
        last_log_line = line
        if 'Epoch' in line or 'Early stopping' in line or 'HASIL RUN' in line:
            log(f"[{run_tag}] {line}")
            update_status({
                'current_run': run_tag,
                'status': 'training',
                'latest_line': line,
                'elapsed_min': round((time.time() - t0) / 60, 1)
            })

    proc.stdout.close()
    return_code = proc.wait()
    duration_min = round((time.time() - t0) / 60, 2)

    if return_code != 0:
        log(f"Eksperimen {run_tag} gagal dengan exit code {return_code}! Pesan terakhir: {last_log_line}")
        return None

    # Baca hasil
    result_file = ROOT / 'results' / f'eval_{run_tag}.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            metrics = json.load(f)
            metrics['duration_min'] = duration_min
            return metrics
    log(f"Peringatan: berkas evaluasi {result_file} tidak ditemukan.")
    return None

def main_loop():
    log("=== AUTONOMOUS OVERNIGHT TRAINING DAEMON DIMULAI ===")
    log(f"Working Directory: {ROOT}")
    log(f"Log Output       : {LOG_FILE}")
    log(f"Status File      : {STATUS_FILE}")
    log(f"Hentikan kapan saja dengan: touch {STOP_FLAG}")

    best_metrics = load_best_metrics()
    log(f"Baseline performa terbaik saat ini: MAE = {best_metrics['MAE']} thn ({best_metrics.get('run_tag', 'baseline')})")

    experiments = generate_experiments()
    exp_idx = 0
    iteration = 1

    while True:
        if STOP_FLAG.exists():
            log(f"File sinyal henti ({STOP_FLAG}) terdeteksi! Menghentikan loop.")
            STOP_FLAG.unlink(missing_ok=True)
            break

        # Ambil konfigurasi eksperimen
        if exp_idx < len(experiments):
            cfg = experiments[exp_idx]
            exp_idx += 1
            if (ROOT / 'results' / f"eval_{cfg['run_tag']}.json").exists():
                log(f"Eksperimen {cfg['run_tag']} sudah selesai sebelumnya, melewati ke berikutnya.")
                continue
        else:
            # Jika daftar dasar habis, generate variasi acak terarah di sekitar titik terbaik
            iteration += 1
            exp_idx = 0
            rng = random.Random(iteration * 100)
            cfg = {
                'run_tag': f'r6_auto_iter{iteration}',
                'epochs': 35,
                'patience': 7,
                'lr_head': round(rng.uniform(2e-4, 3.5e-4), 5),
                'lr_backbone': round(rng.uniform(6e-5, 1e-4), 5),
                'weight_decay': round(rng.uniform(0.05, 0.10), 3),
                'dropout': round(rng.choice([0.30, 0.35, 0.40, 0.45]), 2),
                'smooth_l1_beta': round(rng.choice([0.7, 0.8, 0.9, 1.0]), 1),
                'aug_glasses_p': round(rng.choice([0.25, 0.30, 0.35, 0.40]), 2),
                'aug_rotate': rng.choice([10, 12, 15]),
                'aug_translate': rng.choice([0.05, 0.06, 0.07]),
                'balanced_sampling': rng.choice([True, False])
            }

        update_status({
            'current_run': cfg['run_tag'],
            'status': 'starting',
            'best_mae_so_far': best_metrics['MAE'],
            'timestamp': datetime.datetime.now().isoformat()
        })

        metrics = run_experiment(cfg)

        if metrics is not None:
            new_mae = metrics['MAE']
            mae_diff = best_metrics['MAE'] - new_mae
            log(f"Hasil {cfg['run_tag']}: MAE={new_mae:.3f} (Best={best_metrics['MAE']:.3f}, Diff={mae_diff:+.3f})")

            if new_mae < best_metrics['MAE']:
                log(f"🔥 PENINGKATAN TERCAPAI! Model baru mengalahkan baseline ({new_mae:.3f} vs {best_metrics['MAE']:.3f})")
                old_metrics = dict(best_metrics)
                best_metrics.update({
                    'MAE': new_mae,
                    'RMSE': metrics.get('RMSE'),
                    'MedAE': metrics.get('MedAE'),
                    'bias': metrics.get('bias'),
                    'R2': metrics.get('R2'),
                    'run_tag': cfg['run_tag'],
                    'updated_at': datetime.datetime.now().isoformat()
                })
                save_best_metrics(best_metrics)

                # Salin model terbaik
                best_ckpt = ROOT / 'runs' / f"M1_s43_{cfg['run_tag']}" / 'best.pt'
                if best_ckpt.exists():
                    shutil.copy2(best_ckpt, ROOT / 'runs' / 'best_model_overall.pt')

                # Catat ke docs/comparison/comparison_results.md
                append_comparison_record(cfg['run_tag'], old_metrics, best_metrics)

                # Buat git commit
                git_commit_improvement(cfg['run_tag'], best_metrics, mae_diff)
            else:
                log(f"Model {cfg['run_tag']} (MAE {new_mae:.3f}) belum melampaui rekor ({best_metrics['MAE']:.3f}). Melanjutkan eksperimen.")

        update_status({
            'last_completed_run': cfg['run_tag'],
            'status': 'idle_between_runs',
            'best_mae_so_far': best_metrics['MAE'],
            'timestamp': datetime.datetime.now().isoformat()
        })

        # Jeda 5 detik sebelum run berikutnya
        time.sleep(5)

    log("=== AUTONOMOUS OVERNIGHT TRAINING DAEMON SELESAI ===")

if __name__ == '__main__':
    main_loop()
