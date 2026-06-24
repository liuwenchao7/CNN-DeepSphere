# CNN-DeepSphere (CNN + DeepSphere) for JUNO Atmospheric Neutrino PID

This repository implements **particle identification (PID)** for JUNO atmospheric neutrino events with a **three-class** setup:

| Class | Label | Physics meaning |
|-------|-------|-----------------|
| `numu` | 0 | mu-like (CC) |
| `nue`  | 1 | e-like (CC) |
| `nc`   | 2 | NC-like |

Two model families are provided:

1. **DeepSphere (fea6 only)** — six per-PMT `elec_fea` features on the HEALPix sphere (`fea/train_fea.py`).
2. **CNN + DeepSphere** — 1D CNN extracts per-PMT waveform features; features are aggregated on the sphere by DeepSphere (`train_cnn_ds_pid.py`).

Reference paper: *Neutrino type identification for atmospheric neutrinos in a large homogeneous liquid scintillation detector* ([arXiv:2503.21353](https://arxiv.org/abs/2503.21353)).

---

## 1. Repository Layout

```
CNN-DeepSphere/
├─ pid_lib/                         # Shared data I/O, splits, normalization, metrics
│  ├─ config.py                       # Class mapping, paths, feature names
│  ├─ data_io.py                      # File discovery, y/y_pmt alignment
│  ├─ splits.py                       # Stratified train/val/test by file id
│  ├─ normalize.py                    # Per-channel z-score (train stats only)
│  ├─ metrics.py                      # Confusion matrix, ROC/AUC, F1
│  └─ waveform_io.py                  # waveform / decon / WFSampling loaders
├─ fea/
│  ├─ train_fea.py                    # 6-feature DeepSphere baseline
│  └─ TRAIN_FEA_TUNABLES.md
├─ WFSampling/
│  ├─ WFSampling_v2.py                # Waveform key-point extraction (v2)
│  ├─ WFSampling_v2.cc                # C++ reference
│  ├─ v1/WFSampling_v1.py             # v1 extractor
│  └─ show.py                         # 3-panel WFS visualization
├─ train_cnn_ds_pid.py                # Main CNN+DeepSphere training script
├─ train_200_fht.py                   # Legacy direction-reconstruction script (archive)
├─ tools/
│  ├─ audit_pid_data.py               # Read-only data audit
│  ├─ visualize_pid_data.py           # det_fea + waveform comparison
│  ├─ visualize_pid_event_suite.py    # Cross-modal event suite
│  ├─ plot_pid_scores_energy.py       # PID score vs visE binned AUC
│  ├─ export_test_vise.py             # Export visE aligned to test predictions
│  ├─ preprocess_pid_fht_window.py     # FHT-window waveform crop
│  └─ convert_wfs_npy_compat.py       # numpy2 → list format for deepsphere env
├─ docs/
│  ├─ EXECUTION_PLAN.md               # Task list and acceptance criteria
│  ├─ FULL_EXPERIMENT_STATUS.md        # Experiment progress, shapes, timing
│  ├─ DATA_AND_VISUALIZATION.md       # Data format and plot interpretation
│  ├─ SCORE_AND_BOOTSTRAP.md          # Score/AUC plotting conventions
│  └─ PAPER_PID_NOTES.md              # Paper method summary
├─ audit/                             # Audit JSON and example plots
├─ outputs/                           # Experiment outputs (metrics, manifests; no checkpoints)
├─ execution_plan/                    # Dated task briefs
└─ executioin_history/                # Agent execution logs
```

---

## 2. Method Overview

### 2.1 DeepSphere baseline (fea6)

**Input:** six per-PMT electronic features from `elec_fea/`:

- `x_fht_pmt`, `x_npe_pmt`, `x_nperatio4_pmt`
- `x_peak_pmt`, `x_peaktime_pmt`, `x_slope4_pmt`

**Processing:**

1. Map PMT hits to HEALPix pixels.
2. Z-score normalize using **training-set** statistics only.
3. PMTs with `fht <= 0` or `peaktime >= 1008` are treated as no-hit (`fill_nan = -3.0`).
4. DeepSphere graph network → 3-class softmax.

### 2.2 CNN + DeepSphere

**Input:** fea6 **+** one waveform source per experiment:

| Waveform source | CLI flag | Description |
|-----------------|----------|-------------|
| raw waveform | `--waveform-source waveform` | Original PMT waveform |
| decon waveform | `--waveform-source decon_waveform` | Deconvolved waveform |
| decon npe vs t | `--waveform-source decon_npevst` | Decon NPE-vs-time |
| WFSampling (raw) | `--waveform-source wfsampling` | Key points from raw waveform |
| WFSampling (decon) | `--waveform-source wfs_decon_waveform` | Key points from decon waveform |

**Processing:**

1. 1D CNN on each PMT time series → compact feature vector.
2. Concatenate with fea6 (or use waveform-only mode).
3. DeepSphere aggregation → 3-class output.

WFSampling v2 parameters: `thr_fht=0.2`; `global_thr=15` (raw) / `1` (decon).

---

## 3. Data Requirements

### 3.1 Raw NPZ data (on cluster; not shipped in this repo)

| Class | Path |
|-------|------|
| numu | `/disk_pool1/weijsh/waveform/npy` |
| nue  | `/disk_pool1/liuxy/nu_e/npy` |
| nc   | `/disk_pool1/liuxk/Muon/J24/nc/npy` |

Each class directory contains `det_fea/`, `elec_fea/`, `waveform/`, `y/`.

### 3.2 WFSampling outputs

| Version | Path |
|---------|------|
| v1 raw | `/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v1/{numu,nue,nc}/` |
| v2 raw | `/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2/{numu,nue,nc}/` |
| v2 decon | `/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2/{numu,nue,nc}/` |

Format: `wfsampling_*.npy` (dict with `time/adc/offsets` per event). For the `deepsphere` conda env, run `tools/convert_wfs_npy_compat.py` if files were saved with numpy 2.x.

### 3.3 Labels

**Class labels come from the directory name**, not from `y[:,4]` (PDG code). NC events may contain both ±12 and ±14 in `y[:,4]`; this is expected.

---

## 4. Environment Setup

| Task | Conda env | Notes |
|------|-----------|-------|
| DeepSphere / CNN+DS training | `deepsphere` | Python 3.6, TF 2.6.2 |
| WFSampling v2 preprocessing | `cnn_deepsphere` | numpy 2.x OK |
| Plotting (some tools) | `cnn_deepsphere` | matplotlib, sklearn |

```bash
source /disk_pool1/liuwc/anaconda3/etc/profile.d/conda.sh
conda activate deepsphere   # training
# conda activate cnn_deepsphere  # WFSampling / plotting
```

> Update hard-coded paths in `pid_lib/config.py` if your data lives elsewhere.

---

## 5. Training Tutorial

### 5.1 Data audit (read-only)

```bash
cd /path/to/CNN-DeepSphere
python3 tools/audit_pid_data.py --output-dir audit/
```

### 5.2 DeepSphere fea6 baseline

```bash
conda activate deepsphere
python3 fea/train_fea.py --output-dir outputs/fea6_smoke --gpu 0 --smoke-test

# Full training (background)
nohup env CUDA_VISIBLE_DEVICES=0 python3 fea/train_fea.py \
  --output-dir outputs/fea6 --gpu 0 > logs/train_fea6.log 2>&1 &
```

### 5.3 CNN + DeepSphere (example: fea6 + WFSampling v2)

```bash
conda activate deepsphere

# Smoke test
python3 train_cnn_ds_pid.py \
  --output-dir outputs/fea6+wfs_wav_v2_smoke \
  --waveform-source wfsampling \
  --wfsampling-root /disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2 \
  --gpu 0 --smoke-test

# Full training
nohup env CUDA_VISIBLE_DEVICES=3 python3 -u train_cnn_ds_pid.py \
  --output-dir outputs/fea6+wfs_wav_v2 \
  --waveform-source wfsampling \
  --batch-size 16 --pmt-batch 200 --shuffle-buffer 0 \
  --gpu 3 > logs/train_cnn_ds_wfs_wav_v2_full.log 2>&1 &
```

Common arguments:

- `--waveform-source` — see table in Section 2.2
- `--batch-size 16`, `--pmt-batch 200` — defaults tuned for 17612 PMTs
- `--shuffle-buffer 0` — shuffle event index each epoch (recommended for large waveforms)
- `--gpu N` — physical GPU id

Outputs under `--output-dir`:

- `class_mapping.json`, `manifest_*.json`, `norm_stats.json`
- `loss_log.txt`, `test_metrics.json`
- `checkpoint_pid` (not tracked in git; regenerate by training)

### 5.4 WFSampling v2 preprocessing

```bash
conda activate cnn_deepsphere
nohup python3 -u WFSampling/WFSampling_v2.py --class numu --workers 8 \
  > logs/wfs_v2_numu.log 2>&1 &
# Repeat for nue, nc; use --decon for decon waveform branch
```

### 5.5 Plot PID scores vs visible energy

```bash
/disk_pool1/liuwc/anaconda3/envs/cnn_deepsphere/bin/python3 tools/plot_pid_scores_energy.py \
  --model-dir outputs/fea6 --model-label fea6 \
  --model-dir outputs/fea6+wfs_wav --model-label fea6+wfs_wav \
  --output-dir audit/plots/wfs_wav
```

See `docs/SCORE_AND_BOOTSTRAP.md` for binning and bootstrap conventions.

---

## 6. Experiment Status

Current experiment matrix, input shapes, and per-epoch timing are documented in:

```bash
cat docs/FULL_EXPERIMENT_STATUS.md
```

Completed baselines (examples):

| Run | Test accuracy |
|-----|---------------|
| `outputs/fea6` | see `test_metrics.json` |
| `outputs/fea6+wfs_wav` (v1 WFS) | ~0.759 |

---

## 7. FAQ / Troubleshooting

### Q1: `np.load` fails on WFS `.npy` in `deepsphere` env
v2 files may be saved with numpy 2.x. Run:

```bash
python3 tools/convert_wfs_npy_compat.py --root /path/to/WFS_wav_v2
```

### Q2: Training is slow to start
Building the event index takes ~10–15 minutes before GPU utilization appears.

### Q3: `decon_npevst` is extremely slow (~33 h/epoch)
Use WFSampling-compressed waveforms (`wfsampling` / `wfs_decon_waveform`) instead; timesteps ≈ 60–90 vs thousands.

### Q4: All predictions collapse to one class
Ensure training uses shuffled events (`shuffle_buffer=0` shuffles index each epoch). See `fea/CHANGES.md`.

### Q5: Hard-coded `/disk_pool1/...` paths
Edit `pid_lib/config.py` and CLI `--wfsampling-root` / `--wfs-decon-root` for your environment.

---

## 8. Acknowledgements

- DeepSphere graph neural network on HEALPix sphere.
- WFSampling key-point extraction adapted from the JUNO waveform analysis chain.
- Project agent notes: `AGENTS.md`, execution history under `executioin_history/`.

---

## License

No LICENSE file is included yet. Add one before public distribution.
