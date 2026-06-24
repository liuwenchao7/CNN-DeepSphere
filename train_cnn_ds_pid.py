#!/usr/bin/env python3
"""CNN + DeepSphere PID training (direction-script style backbone)."""

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--waveform-source",
        required=True,
        choices=[
            "decon_npevst",
            "decon_waveform",
            "waveform",
            "wfsampling",
            "wfs_decon_waveform",
            "decon_waveform_fht",
            "waveform_fht",
        ],
    )
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-class", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--fill-nan", type=float, default=-3.0)
    p.add_argument("--wfs-decon-root",
                   default="/disk_pool1/liuwc/data/cnn+ds/pid/WFS_decon_wav_v2")
    p.add_argument("--wfsampling-root", default="/disk_pool1/liuwc/data/cnn+ds/pid/WFS_wav_v2")
    p.add_argument("--fht-window-start", type=float, default=-20.0)
    p.add_argument("--fht-window-width", type=int, default=300)
    p.add_argument("--decon-fht-root", default="/disk_pool1/liuwc/data/cnn+ds/pid/decon_wav_fht+-/wav")
    p.add_argument("--waveform-fht-root", default="/disk_pool1/liuwc/data/cnn+ds/pid/wav_fht+-/wav")
    p.add_argument("--cache-files", type=int, default=8)
    p.add_argument("--pmt-batch-size", type=int, default=200,
                   help="PMTs processed per CNN sub-batch (lower avoids GPU OOM)")
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="tf.data shuffle buffer size; 0 = shuffle event index per epoch only "
        "(avoids slow buffer fill on disk-heavy generators)",
    )
    return p.parse_args()


def setup_tf(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    print(f"[GPU] index={gpu}  visible={tf.config.list_physical_devices('GPU')}")
    return tf


def _cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(cache: OrderedDict, key, val, max_items: int):
    cache[key] = val
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _load_compact_npz_arrays(path):
    """Load compact FHT npz into plain arrays (safe for LRU cache)."""
    z = np.load(path, allow_pickle=False)
    return {
        "event_offsets": np.asarray(z["event_offsets"]).copy(),
        "copyNo": np.asarray(z["copyNo"]).copy(),
        "waveform": np.asarray(z["waveform"]).copy(),
    }


def _event_from_compact_z(z, event_idx, max_len, invert=False):
    from pid_lib.config import N_PMT
    evt_off = int(z["event_offsets"][event_idx])
    evt_end = int(z["event_offsets"][event_idx + 1])
    copy = z["copyNo"][evt_off:evt_end].astype(np.int64)
    wf = z["waveform"][evt_off:evt_end].astype(np.float32)
    if invert:
        wf = 16384.0 - wf
    out = np.zeros((N_PMT, max_len, 1), dtype=np.float32)
    n = min(max_len, wf.shape[1])
    if len(copy):
        out[copy, :n, 0] = wf[:, :n]
    return out


def _load_compact_fixed_wave(path, event_idx, max_len, invert=False):
    z = np.load(path, allow_pickle=False)
    return _event_from_compact_z(z, event_idx, max_len, invert=invert)


def make_sample_loader(args, norm_stats, max_wf_len, max_wf_pts):
    from pid_lib.data_io import find_waveform_files, load_aligned_file, mark_no_hit_features
    from pid_lib.normalize import apply_norm
    from pid_lib.waveform_io import (
        decon_npevst_event_from_file,
        load_decon_waveform,
        load_waveform,
        load_wfsampling_file,
        wfsampling_event_from_file,
    )

    src = args.waveform_source
    fill_nan = args.fill_nan
    ws_root = (
        args.wfs_decon_root if src == "wfs_decon_waveform" else args.wfsampling_root
    )
    file_cache = OrderedDict()
    wave_cache = OrderedDict()
    max_cache = max(1, int(args.cache_files))

    def load_elec(entry):
        key = (entry.cls_name, entry.file_id)
        cached = _cache_get(file_cache, key)
        if cached is not None:
            return cached
        x_elec, _, meta = load_aligned_file(entry)
        x_elec, _ = mark_no_hit_features(x_elec)
        x_elec = apply_norm(x_elec, norm_stats, fill_nan=fill_nan).astype(np.float32)
        n_kept = int(meta.get("n_kept", x_elec.shape[0]))
        _cache_put(file_cache, key, (x_elec, n_kept), max_cache)
        return x_elec, n_kept

    def load_event(entry, event_idx):
        x_elec, n_kept = load_elec(entry)
        if event_idx >= n_kept:
            raise IndexError(f"event_idx={event_idx} out of range n_kept={n_kept}")
        fea = x_elec[event_idx]

        if src == "decon_npevst":
            key = ("decon_npevst", entry.cls_name, entry.file_id)
            file_arr = _cache_get(wave_cache, key)
            if file_arr is None:
                paths = find_waveform_files(entry.root, entry.file_id)
                path = paths.get("decon_npevst")
                if not path:
                    raise FileNotFoundError(f"decon_npevst missing for {entry.cls_name} {entry.file_id}")
                file_arr = np.load(path, allow_pickle=True)
                _cache_put(wave_cache, key, file_arr, max_cache)
            wave, _ = decon_npevst_event_from_file(file_arr, event_idx, max_wf_pts)
        elif src == "decon_waveform":
            wave = load_decon_waveform(entry, event_idx, max_len=max_wf_len)
        elif src == "waveform":
            wave = load_waveform(entry, event_idx, max_len=max_wf_len)
        elif src in ("wfsampling", "wfs_decon_waveform"):
            key = (src, entry.cls_name, entry.file_id)
            cached = _cache_get(wave_cache, key)
            if cached is None:
                cached = load_wfsampling_file(entry, ws_root=ws_root)
                _cache_put(wave_cache, key, cached, max_cache)
            wave, _ = wfsampling_event_from_file(cached, event_idx, max_wf_pts)
        elif src == "decon_waveform_fht":
            key = ("decon_fht", entry.cls_name, entry.file_id)
            z = _cache_get(wave_cache, key)
            if z is None:
                path = os.path.join(
                    args.decon_fht_root, entry.cls_name,
                    f"decon_waveform_{entry.file_id}.npz",
                )
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                z = _load_compact_npz_arrays(path)
                _cache_put(wave_cache, key, z, max_cache)
            wave = _event_from_compact_z(z, event_idx, max_wf_len, invert=False)
        else:
            key = ("waveform_fht", entry.cls_name, entry.file_id)
            z = _cache_get(wave_cache, key)
            if z is None:
                path = os.path.join(
                    args.waveform_fht_root, entry.cls_name,
                    f"waveform_{entry.file_id}.npz",
                )
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                z = _load_compact_npz_arrays(path)
                _cache_put(wave_cache, key, z, max_cache)
            # Preprocess already inverts raw waveform (16384-ADC); do not invert again.
            wave = _event_from_compact_z(z, event_idx, max_wf_len, invert=False)
        return wave.astype(np.float32), fea.astype(np.float32), int(entry.label)

    return load_event, load_elec


def build_event_index(entries, load_elec_fn, balance_events=False, seed=42):
    """Build a fixed (entry, event_idx) list once for deterministic epochs."""
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    index = []
    skipped = 0
    totals = defaultdict(int)

    for entry in shuffled:
        try:
            _, n_kept = load_elec_fn(entry)
        except Exception:
            skipped += 1
            continue
        for evt in range(int(n_kept)):
            index.append((entry, evt))
            totals[entry.label] += 1

    if balance_events and index:
        by_label = defaultdict(list)
        for entry, evt in index:
            by_label[entry.label].append((entry, evt))
        cap = min(len(v) for v in by_label.values())
        index = []
        for label in sorted(by_label.keys()):
            index.extend(by_label[label][:cap])
        random.Random(seed + 1).shuffle(index)
        print(f"[balance] event totals={dict(totals)} cap_per_class={cap}")
    else:
        print(f"[index] events={len(index)} totals={dict(totals)} skipped_files={skipped}")

    return index


def make_indexed_generator(event_index, load_event_fn, shuffle_epochs=False, seed=42):
    """Yield samples from a fixed index; optionally reshuffle index each epoch."""
    epoch_no = 0

    def _gen():
        nonlocal epoch_no
        idx = list(event_index)
        if shuffle_epochs:
            random.Random(seed + epoch_no).shuffle(idx)
        epoch_no += 1
        dropped = 0
        for entry, evt in idx:
            try:
                wave, fea, label = load_event_fn(entry, evt)
                yield (wave, fea), label
            except Exception:
                dropped += 1
                continue
        if dropped:
            print(f"[warn] generator dropped {dropped}/{len(event_index)} indexed events")

    return _gen


def build_dataset(tf, event_index, load_event_fn, load_elec_fn, wf_shape, feat_num, batch_size,
                  shuffle=False, seed=42, shuffle_buffer=0):
    from pid_lib.config import N_PMT

    out_sig = (
        (
            tf.TensorSpec(shape=wf_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(N_PMT, feat_num), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    gen_fn = make_indexed_generator(
        event_index, load_event_fn, shuffle_epochs=shuffle, seed=seed
    )
    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=out_sig)
    if shuffle_buffer > 0:
        cap = min(shuffle_buffer, max(len(event_index), 1))
        ds = ds.shuffle(buffer_size=cap, seed=seed, reshuffle_each_iteration=False)
        print(f"[data] tf.data shuffle buffer={cap}")
    elif shuffle:
        print("[data] per-epoch index shuffle (shuffle_buffer=0)")
    opts = tf.data.Options()
    opts.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return ds.with_options(opts).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn_model(tf, timesteps, channels, static_feats, feature_num):
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
    from tensorflow.keras.models import Model

    model_input = Input(shape=(timesteps + static_feats, channels))
    static_part = model_input[:, :static_feats, :]
    dynamic_part = model_input[:, static_feats:, :]

    x = Conv1D(8, kernel_size=3, activation="relu", padding="same")(dynamic_part)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(16, kernel_size=3, activation="relu", padding="same")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(feature_num)(x)

    # static features are duplicated across channels when channels>1, keep one copy.
    static_flat = Flatten()(static_part[:, :, :1])
    merged = Concatenate(axis=-1)([x, static_flat])
    return Model(model_input, merged)


def pmt_batch_cnn(tf, model_0, model_1, input_0, input_1, timesteps, channels, out_dim, pmt_batch_size):
    b = tf.shape(input_0)[0]
    n_pmt = input_0.shape[1]

    hama_features = tf.zeros((b, tf.shape(input_0)[1], out_dim), dtype=tf.float32)
    for i in range(0, n_pmt, pmt_batch_size):
        end_idx = min(i + pmt_batch_size, n_pmt)
        one = input_0[:, i:end_idx, :, :]
        one = tf.reshape(one, (-1, timesteps, channels))
        feat = model_0(one)
        feat = tf.reshape(feat, (b, end_idx - i, out_dim))
        left = hama_features[:, :i, :]
        right = hama_features[:, end_idx:, :]
        hama_features = tf.concat([left, feat, right], axis=1)

    nnvt_features = tf.zeros((b, tf.shape(input_1)[1], out_dim), dtype=tf.float32)
    for i in range(0, input_1.shape[1], pmt_batch_size):
        end_idx = min(i + pmt_batch_size, input_1.shape[1])
        one = input_1[:, i:end_idx, :, :]
        one = tf.reshape(one, (-1, timesteps, channels))
        feat = model_1(one)
        feat = tf.reshape(feat, (b, end_idx - i, out_dim))
        left = nnvt_features[:, :i, :]
        right = nnvt_features[:, end_idx:, :]
        nnvt_features = tf.concat([left, feat, right], axis=1)

    return hama_features, nnvt_features


def build_model(tf, wf_shape, feat_num, nside, indices, which_pixel, count_pix, list_0, list_1, pmt_batch_size):
    from tensorflow.keras.layers import Input, Lambda, Flatten, Dense
    from tensorflow.keras.models import Model
    from deepsphere import HealpyGCNN
    from deepsphere import healpy_layers as hp_layer
    from pid_lib.config import N_PMT

    static_feats = 6
    feature_num = 6
    merge_dim = static_feats + feature_num  # 12
    npix = count_pix.shape[1]
    timesteps, channels = wf_shape[1], wf_shape[2]

    wave_input = Input(shape=(N_PMT, timesteps, channels), name="waveform")
    fea_input = Input(shape=(N_PMT, feat_num), name="elec_features")

    # Optional "group_input" style merge: put static 6 features before waveform time axis.
    fea_as_time = Lambda(lambda x: tf.expand_dims(x, axis=-1), name="fea_to_time")(fea_input)  # (B, N_PMT, 6, 1)
    if channels > 1:
        fea_as_time = Lambda(lambda x: tf.tile(x, [1, 1, 1, channels]), name="fea_tile_channels")(fea_as_time)
    group_input = Lambda(lambda x: tf.concat([x[0], x[1]], axis=2), name="group_input")([fea_as_time, wave_input])

    cnn0 = build_cnn_model(tf, timesteps, channels, static_feats=static_feats, feature_num=feature_num)
    cnn1 = build_cnn_model(tf, timesteps, channels, static_feats=static_feats, feature_num=feature_num)
    class PMTBatchCNNLayer(tf.keras.layers.Layer):
        def __init__(self, cnn_model_0, cnn_model_1, timesteps_, channels_, out_dim_, pmt_batch_size_, **kwargs):
            super().__init__(**kwargs)
            self.cnn_model_0 = cnn_model_0
            self.cnn_model_1 = cnn_model_1
            self.timesteps_ = timesteps_
            self.channels_ = channels_
            self.out_dim_ = out_dim_
            self.pmt_batch_size_ = pmt_batch_size_

        def call(self, inputs):
            return pmt_batch_cnn(
                tf,
                self.cnn_model_0,
                self.cnn_model_1,
                inputs[0],
                inputs[1],
                self.timesteps_,
                self.channels_,
                self.out_dim_,
                self.pmt_batch_size_,
            )

    pmt_layer = PMTBatchCNNLayer(
        cnn0, cnn1, timesteps + static_feats, channels, merge_dim, pmt_batch_size
    )

    hama_in = Lambda(lambda x: tf.gather(x, list_0, axis=1), name="hama_in")(group_input)
    nnvt_in = Lambda(lambda x: tf.gather(x, list_1, axis=1), name="nnvt_in")(group_input)
    hama_feat, nnvt_feat = pmt_layer([hama_in, nnvt_in])

    def scatter_results(inputs):
        arr0, arr1 = inputs
        b = tf.shape(arr0)[0]
        out = tf.zeros((b, N_PMT, merge_dim), dtype=arr0.dtype)
        bi = tf.range(b)[:, tf.newaxis]
        bi0 = tf.tile(bi, [1, len(list_0)])
        bi1 = tf.tile(bi, [1, len(list_1)])
        idx0 = tf.stack([bi0, tf.tile(tf.expand_dims(tf.constant(list_0, dtype=tf.int32), 0), [b, 1])], axis=-1)
        idx1 = tf.stack([bi1, tf.tile(tf.expand_dims(tf.constant(list_1, dtype=tf.int32), 0), [b, 1])], axis=-1)
        out = tf.tensor_scatter_nd_update(out, idx0, arr0)
        out = tf.tensor_scatter_nd_update(out, idx1, arr1)
        return out

    merged = Lambda(scatter_results, name="scatter_back")([hama_feat, nnvt_feat])

    def pmt_to_pix(inp):
        b = tf.shape(inp)[0]
        pix = tf.zeros((b, npix, merge_dim), dtype=inp.dtype)
        bi = tf.range(b)[:, tf.newaxis]
        bi = tf.tile(bi, [1, N_PMT])
        idx = tf.stack([bi, tf.tile(tf.expand_dims(which_pixel, 0), [b, 1])], axis=-1)
        pix = tf.tensor_scatter_nd_add(pix, idx, inp)
        return pix / count_pix

    pixel_feat = Lambda(pmt_to_pix, name="pmt_to_pix")(merged)

    ds_layers = [
        hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=6, use_bias=True, use_bn=True, activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=3),
        Flatten(),
        Dense(3),
    ]
    ds_model = HealpyGCNN(nside=nside, indices=indices, layers=ds_layers, n_neighbors=40)
    ds_model.build(input_shape=(None, len(indices), merge_dim))
    logits = ds_model(pixel_feat)
    model = Model(inputs=[wave_input, fea_input], outputs=logits)
    return model, merge_dim


def main():
    args = parse_args()
    tf = setup_tf(args.gpu)

    import healpy as hp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    from pid_lib.config import CLASS_LABEL, FEATURE_NAMES, NSIDE, N_PMT, PMT_TYPE_CSV, WHICH_PIXEL_PATH
    from pid_lib.data_io import find_waveform_files
    from pid_lib.metrics import evaluate_predictions, save_class_mapping
    from pid_lib.normalize import estimate_norm_stats, load_norm_stats, save_norm_stats
    from pid_lib.splits import build_or_load_splits, load_manifest
    from pid_lib.waveform_io import estimate_waveform_params

    os.makedirs(args.output_dir, exist_ok=True)
    save_class_mapping(args.output_dir)

    max_files = 3 if args.smoke_test else args.max_files_per_class
    manifest_dir = args.manifest_dir or args.output_dir
    reuse = (args.manifest_dir is not None) and not args.smoke_test
    if reuse and os.path.isfile(os.path.join(manifest_dir, "manifest_train.json")):
        train_e = load_manifest(os.path.join(manifest_dir, "manifest_train.json"))
        val_e = load_manifest(os.path.join(manifest_dir, "manifest_val.json"))
        test_e = load_manifest(os.path.join(manifest_dir, "manifest_test.json"))
        print(f"[splits] Reused from {manifest_dir}")
    else:
        train_e, val_e, test_e = build_or_load_splits(
            args.output_dir, seed=args.seed, max_files_per_class=max_files, reuse=not args.smoke_test
        )
    print(f"Split (files): train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    norm_path = os.path.join(args.output_dir, "norm_stats.json")
    if os.path.isfile(norm_path):
        norm_stats = load_norm_stats(norm_path)
        print("[norm] Loaded existing norm_stats.json")
    elif args.manifest_dir and os.path.isfile(os.path.join(args.manifest_dir, "norm_stats.json")):
        norm_stats = load_norm_stats(os.path.join(args.manifest_dir, "norm_stats.json"))
        print(f"[norm] Loaded from {args.manifest_dir}")
    else:
        norm_stats = estimate_norm_stats(train_e)
        save_norm_stats(args.output_dir, norm_stats)

    src = args.waveform_source
    ws_root = (
        args.wfs_decon_root if src == "wfs_decon_waveform" else args.wfsampling_root
    )
    src_for_est = src
    if src in ("decon_waveform_fht", "waveform_fht"):
        src_for_est = "decon_waveform" if src == "decon_waveform_fht" else "waveform"
    elif src == "wfs_decon_waveform":
        src_for_est = "wfsampling"
    wp = estimate_waveform_params(train_e[:10], src_for_est, ws_root=ws_root)
    if src in ("decon_npevst", "wfsampling", "wfs_decon_waveform"):
        timesteps, channels = wp["max_points"], 2
    elif src in ("decon_waveform_fht", "waveform_fht"):
        timesteps, channels = args.fht_window_width, 1
    else:
        timesteps, channels = wp["max_len"], 1
    max_wf_len = timesteps if src not in ("decon_npevst", "wfsampling", "wfs_decon_waveform") else 0
    max_wf_pts = timesteps if src in ("decon_npevst", "wfsampling", "wfs_decon_waveform") else 0
    print(f"Waveform: source={src} timesteps={timesteps} channels={channels} pmt_batch={args.pmt_batch_size}")

    wf_shape = (N_PMT, timesteps, channels)
    feat_num = len(FEATURE_NAMES)
    load_event_fn, load_elec_fn = make_sample_loader(args, norm_stats, max_wf_len, max_wf_pts)

    batch = args.batch_size
    train_index = build_event_index(
        train_e, load_elec_fn, balance_events=True, seed=args.seed
    )
    val_index = build_event_index(
        val_e, load_elec_fn, balance_events=False, seed=args.seed
    )
    test_index = build_event_index(
        test_e, load_elec_fn, balance_events=False, seed=args.seed
    )

    train_ds = build_dataset(
        tf, train_index, load_event_fn, load_elec_fn, wf_shape, feat_num, batch,
        shuffle=True, seed=args.seed, shuffle_buffer=args.shuffle_buffer,
    )
    val_ds = build_dataset(
        tf, val_index, load_event_fn, load_elec_fn, wf_shape, feat_num, batch, shuffle=False,
    )
    test_ds = build_dataset(
        tf, test_index, load_event_fn, load_elec_fn, wf_shape, feat_num, batch, shuffle=False,
    )

    wp_arr = np.load(WHICH_PIXEL_PATH)
    which_pixel = tf.cast(wp_arr[:, 1], tf.int32)
    df = pd.read_csv(PMT_TYPE_CSV, sep=" ")
    list_0 = df[df["type"] == "Hamamatsu"]["index"].tolist()
    list_1 = df[df["type"].isin(["HighQENNVT", "NNVT"])]["index"].tolist()
    indices = np.arange(hp.nside2npix(NSIDE))
    npix = len(indices)
    count_pix = np.zeros(npix)
    for i in range(N_PMT):
        count_pix[int(wp_arr[i, 1])] += 1
    count_pix[count_pix == 0] = 1
    count_pix = tf.constant(count_pix.reshape(1, npix, 1).astype(np.float32))

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model, merge_dim = build_model(
            tf, wf_shape, feat_num, NSIDE, indices, which_pixel, count_pix, list_0, list_1,
            args.pmt_batch_size,
        )
        model.summary(110)
        print(f"[model] DeepSphere input shape target: {(None, len(indices), merge_dim)}")
        print(f"[model] Output shape: {model.output_shape}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    tracked = [v.name for v in model.trainable_variables if ("conv1d" in v.name.lower() or "dense" in v.name.lower())]
    print("[model] CNN/Dense trainable vars (first 16):")
    for name in tracked[:16]:
        print(f"  {name}")

    ckpt = os.path.join(args.output_dir, "checkpoint_pid")
    if (not args.smoke_test) and os.path.isfile(ckpt + ".index"):
        model.load_weights(ckpt)
        print("[ckpt] Loaded existing checkpoint.")

    class LrDecay(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            old = self.model.optimizer.learning_rate.read_value()
            self.model.optimizer.learning_rate.assign(old * 0.995)

    class LossLogger(tf.keras.callbacks.Callback):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def on_epoch_end(self, epoch, logs=None):
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch {epoch + 1}: "
                    f"loss={logs.get('loss'):.4f}  "
                    f"acc={logs.get('accuracy'):.4f}  "
                    f"val_loss={logs.get('val_loss', float('nan')):.4f}  "
                    f"val_acc={logs.get('val_accuracy', float('nan')):.4f}\n"
                )

    callbacks = [
        EarlyStopping("val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        LrDecay(),
        LossLogger(os.path.join(args.output_dir, "loss_log.txt")),
    ]

    epochs = args.smoke_epochs if args.smoke_test else args.epochs
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    elapsed = time.time() - t0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="val")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "learning_curve.png"), dpi=200)
    plt.close()

    y_logits_list, y_true_list = [], []
    for (wave_b, fea_b), yb in test_ds:
        pred = model([wave_b, fea_b], training=False)
        y_logits_list.append(pred.numpy())
        y_true_list.append(yb.numpy())
    y_logits = np.concatenate(y_logits_list)
    y_true = np.concatenate(y_true_list)
    y_prob = tf.nn.softmax(y_logits).numpy()
    metrics = evaluate_predictions(y_true, y_prob, args.output_dir)

    _h, _r = divmod(elapsed, 3600)
    _m, _s = divmod(_r, 60)
    with open(os.path.join(args.output_dir, "detail.txt"), "w", encoding="utf-8") as f:
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"command: {' '.join(sys.argv)}\n")
        f.write(f"waveform_source: {src}\n")
        f.write(f"waveform_shape: {wf_shape}  (N_PMT, timesteps, channels)\n")
        f.write(f"deepsphere_input_shape: {(None, len(indices), merge_dim)}\n")
        f.write(f"model_output_shape: {model.output_shape}\n")
        f.write(f"gpu: {args.gpu}\n")
        f.write(f"class_mapping: {json.dumps(CLASS_LABEL)}\n")
        f.write(f"feature_order: {FEATURE_NAMES}\n")
        f.write(f"fill_nan: {args.fill_nan}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"batch_size: {batch}\n")
        f.write(f"pmt_batch_size: {args.pmt_batch_size}\n")
        f.write(f"shuffle_buffer: {args.shuffle_buffer}\n")
        f.write(f"epochs_trained: {len(history.history['loss'])}\n")
        f.write(f"test_accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"test_macro_auc: {metrics['macro_auc']:.6f}\n")
        f.write(f"elapsed: {int(_h)}h {int(_m)}m {int(_s)}s\n")

    print(f"\n[done] Test accuracy: {metrics['accuracy']:.4f}  Macro AUC: {metrics['macro_auc']:.4f}")
    print(json.dumps(metrics["per_class"], indent=2))


if __name__ == "__main__":
    main()
