#!/usr/bin/env python3
"""CNN + DeepSphere PID for JUNO atmospheric neutrinos (3 waveform ablations)."""


import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--waveform-source", choices=["decon_npevst", "decon_waveform", "wfsampling"], required=True)
    p.add_argument("--manifest-dir", default=None, help="Reuse file splits from fea baseline manifests")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-class", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--wfsampling-root", default=None)
    return p.parse_args()


def setup_tf(gpu: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf

    print("GPU", gpu, tf.config.list_physical_devices("GPU"))
    return tf


def main():
    args = parse_args()
    tf = setup_tf(args.gpu)

    import numpy as np
    import pandas as pd
    import healpy as hp
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, Flatten, Dense, Lambda, Concatenate,
    )
    from tensorflow.keras.models import Model
    from deepsphere import HealpyGCNN
    from deepsphere import healpy_layers as hp_layer

    from pid_lib.config import (
        CLASS_LABEL, FEATURE_NAMES, NSIDE, N_PMT, PMT_TYPE_CSV, WHICH_PIXEL_PATH,
        WFSAMPLING_OUTPUT_ROOT,
    )
    from pid_lib.data_io import load_aligned_file, mark_no_hit_features, FileEntry
    from pid_lib.metrics import compute_class_weights, evaluate_predictions, save_class_mapping
    from pid_lib.normalize import apply_norm, estimate_norm_stats, load_norm_stats, save_norm_stats
    from pid_lib.splits import build_or_load_splits, load_manifest
    from pid_lib.waveform_io import (
        estimate_waveform_params,
        load_decon_npevst,
        load_decon_waveform,
        load_wfsampling,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_class_mapping(args.output_dir)
    ws_root = args.wfsampling_root or WFSAMPLING_OUTPUT_ROOT

    manifest_dir = args.manifest_dir or args.output_dir
    max_files = 2 if args.smoke_test else args.max_files_per_class
    if args.manifest_dir and os.path.isfile(os.path.join(manifest_dir, "manifest_train.json")):
        train_entries = load_manifest(os.path.join(manifest_dir, "manifest_train.json"))
        val_entries = load_manifest(os.path.join(manifest_dir, "manifest_val.json"))
        test_entries = load_manifest(os.path.join(manifest_dir, "manifest_test.json"))
    else:
        train_entries, val_entries, test_entries = build_or_load_splits(
            manifest_dir, seed=args.seed, max_files_per_class=max_files, reuse=not args.smoke_test
        )

    norm_path = os.path.join(args.output_dir, "norm_stats.json")
    if os.path.isfile(norm_path):
        norm_stats = load_norm_stats(norm_path)
    else:
        norm_stats = estimate_norm_stats(train_entries)
        save_norm_stats(args.output_dir, norm_stats)

    wave_params = estimate_waveform_params(train_entries, args.waveform_source, ws_root=ws_root)
    with open(os.path.join(args.output_dir, "waveform_params.json"), "w", encoding="utf-8") as f:
        json.dump({"source": args.waveform_source, **wave_params}, f, indent=2)

    if args.waveform_source == "decon_waveform":
        timesteps = wave_params["max_len"]
        channels = 1
    elif args.waveform_source == "decon_npevst":
        timesteps = wave_params["n_bins"]
        channels = 1
    else:
        timesteps = wave_params["max_points"]
        channels = 2

    static_feats = len(FEATURE_NAMES)
    elec_dim = static_feats
    cnn_out_dim = 4
    merge_dim = cnn_out_dim + elec_dim

    which_pixel_np = np.load(WHICH_PIXEL_PATH)
    which_pixel = tf.cast(which_pixel_np[:, 1], tf.int32)
    df = pd.read_csv(PMT_TYPE_CSV, sep=" ")
    list_0 = df[df["type"] == "Hamamatsu"]["index"].tolist()
    list_1 = df[df["type"].isin(["HighQENNVT", "NNVT"])]["index"].tolist()
    indices = np.arange(hp.nside2npix(NSIDE))
    npix = len(indices)
    count_pix = np.zeros(npix)
    for i in range(N_PMT):
        count_pix[int(which_pixel_np[i, 1])] += 1
    count_pix[count_pix == 0] = 1
    count_pix = count_pix.reshape(1, npix, 1).astype(np.float32)

    def load_event(entry: FileEntry, event_idx: int):
        x_elec, labels, _ = load_aligned_file(entry)
        x_elec = x_elec[event_idx : event_idx + 1]
        x_elec, _ = mark_no_hit_features(x_elec)
        x_elec = apply_norm(x_elec, norm_stats)[0]
        if args.waveform_source == "decon_waveform":
            wave = load_decon_waveform(entry, event_idx, timesteps)
        elif args.waveform_source == "decon_npevst":
            wave = load_decon_npevst(entry, event_idx, n_bins=timesteps)
        else:
            wave, _ = load_wfsampling(entry, event_idx, timesteps, ws_root=ws_root)
        return wave, x_elec, int(labels[event_idx])

    def gen_entries(entries):
        for entry in entries:
            _, labels, meta = load_aligned_file(entry)
            n = meta.get("n_kept", len(labels))
            for evt in range(n):
                yield entry, evt

    def data_generator(entries):
        for entry, evt in gen_entries(entries):
            wave, fea, lab = load_event(entry, evt)
            yield wave, fea, lab

    out_sig = (
        tf.TensorSpec((N_PMT, timesteps, channels), tf.float32),
        tf.TensorSpec((N_PMT, static_feats), tf.float32),
        tf.TensorSpec((), tf.int32),
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_entries), output_signature=out_sig
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(val_entries), output_signature=out_sig
    )
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_entries), output_signature=out_sig
    )
    train_ds = train_ds.shuffle(2048).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    wave_input = Input(shape=(N_PMT, timesteps, channels), name="waveform")
    fea_input = Input(shape=(N_PMT, static_feats), name="elec_features")

    def build_cnn():
        inp = Input(shape=(timesteps, channels))
        x = Conv1D(8, 3, activation="relu", padding="same")(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(16, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(16, 3, activation="relu", padding="same")(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(cnn_out_dim)(x)
        return Model(inp, x)

    cnn0, cnn1 = build_cnn(), build_cnn()

    def pmt_batch_cnn(cnn_m, pmt_idx_list, wave):
        feats = tf.zeros((tf.shape(wave)[0], len(pmt_idx_list), cnn_out_dim))
        pmt_batch = 500
        for i in range(0, len(pmt_idx_list), pmt_batch):
            end_i = min(i + pmt_batch, len(pmt_idx_list))
            sl = wave[:, pmt_idx_list[i:end_i], :, :]
            b = tf.shape(sl)[0]
            flat = tf.reshape(sl, (-1, timesteps, channels))
            f = cnn_m(flat)
            f = tf.reshape(f, (b, end_i - i, cnn_out_dim))
            left, right = feats[:, :i, :], feats[:, end_i:, :]
            feats = tf.concat([left, f, right], axis=1)
        return feats

    hama_w = Lambda(lambda x: tf.gather(x, list_0, axis=1))(wave_input)
    nnvt_w = Lambda(lambda x: tf.gather(x, list_1, axis=1))(wave_input)
    hama_f = Lambda(lambda x: tf.gather(x, list_0, axis=1))(fea_input)
    nnvt_f = Lambda(lambda x: tf.gather(x, list_1, axis=1))(fea_input)

    hama_cnn = Lambda(lambda w: pmt_batch_cnn(cnn0, list_0, w))(wave_input)
    nnvt_cnn = Lambda(lambda w: pmt_batch_cnn(cnn1, list_1, w))(wave_input)

    def scatter_merge(cnn_part, fea_part, idx_list):
        b = tf.shape(cnn_part)[0]
        full = tf.zeros((b, N_PMT, merge_dim))
        merged = Concatenate(axis=-1)([cnn_part, fea_part])
        bi = tf.range(b)[:, tf.newaxis]
        bi = tf.tile(bi, [1, len(idx_list)])
        ind = tf.stack([bi, tf.tile(tf.expand_dims(tf.constant(idx_list, dtype=tf.int32), 0), [b, 1])], axis=-1)
        return tf.tensor_scatter_nd_update(full, ind, merged)

    hama_m = Lambda(lambda t: scatter_merge(t[0], t[1], list_0))([hama_cnn, hama_f])
    nnvt_m = Lambda(lambda t: scatter_merge(t[0], t[1], list_1))([nnvt_cnn, nnvt_f])
    merged = hama_m + nnvt_m  # disjoint PMT sets

    def pmt_to_pix(inp):
        b = tf.shape(inp)[0]
        pix = tf.zeros((b, npix, merge_dim))
        bi = tf.range(b)[:, tf.newaxis]
        bi = tf.tile(bi, [1, N_PMT])
        ind = tf.stack([bi, tf.tile(tf.expand_dims(which_pixel, 0), [b, 1])], axis=-1)
        pix = tf.tensor_scatter_nd_add(pix, ind, inp)
        return pix / count_pix

    pixel_feat = Lambda(pmt_to_pix)(merged)
    assert merge_dim == cnn_out_dim + elec_dim

    layers = [
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
    ds_model = HealpyGCNN(nside=NSIDE, indices=indices, layers=layers, n_neighbors=40)
    ds_model.build(input_shape=(None, len(indices), merge_dim))
    logits = ds_model(pixel_feat)
    model = Model(inputs=[wave_input, fea_input], outputs=logits)

    y_train = []
    for entry in train_entries:
        _, lab, _ = load_aligned_file(entry)
        y_train.append(lab)
    y_train = np.concatenate(y_train)
    class_weight = compute_class_weights(y_train)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        ckpt = os.path.join(args.output_dir, "checkpoint_pid_cnn_ds")
        callbacks = [
            tf.keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(ckpt, save_weights_only=True, monitor="val_loss", save_best_only=True),
        ]
        epochs = args.smoke_epochs if args.smoke_test else args.epochs
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "learning_curve.png"), dpi=200)
        plt.close()

    y_true, logits_list = [], []
    for wb, fb, yb in test_ds:
        pred = model.predict([wb, fb], verbose=0)
        logits_list.append(pred)
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_prob = tf.nn.softmax(np.concatenate(logits_list)).numpy()
    metrics = evaluate_predictions(y_true, y_prob, args.output_dir)

    with open(os.path.join(args.output_dir, "detail.txt"), "w", encoding="utf-8") as f:
        f.write(f"waveform_source: {args.waveform_source}\n")
        f.write(f"timesteps: {timesteps} channels: {channels}\n")
        f.write(f"merge_dim: {merge_dim}\n")
        f.write(f"command: {' '.join(sys.argv)}\n")
        f.write(f"test_accuracy: {metrics['accuracy']}\n")
        f.write(f"test_macro_auc: {metrics['macro_auc']}\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
