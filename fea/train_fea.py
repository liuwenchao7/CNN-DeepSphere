#!/usr/bin/env python3
"""DeepSphere 6-feature PID baseline for JUNO atmospheric neutrino classification.

Root-cause fix vs. previous run:
  - Previous run used a sequential generator (numu→nue→nc), causing the model
    to learn nc-biased features toward convergence.  This version uses per-class
    tf.data.Dataset objects interleaved with equal sampling weights, guaranteeing
    every batch has a balanced class mix throughout training.
  - fill_nan is now -3.0 (well outside the normalised feature range) so no-hit
    PMTs are cleanly separable from physically zero values.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True,
                   help="e.g. outputs/fea6  (use a meaningful name)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-files-per-class", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="3 files/class, 3 epochs – fast sanity check")
    p.add_argument("--smoke-epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    p.add_argument("--lr-decay", type=float, default=0.995,
                   help="Multiplicative decay applied every epoch end")
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--shuffle-buffer", type=int, default=8192)
    p.add_argument("--fill-nan", type=float, default=-3.0,
                   help="Value for no-hit PMTs after normalisation")
    return p.parse_args()


# ── TF setup ─────────────────────────────────────────────────────────────────
def setup_tf(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[GPU] index={gpu}  visible={gpus}")
    return tf


# ── Per-class generator ──────────────────────────────────────────────────────
def make_flat_generator(entries, norm_stats, fill_nan, seed=None):
    """Yield (x, label) for every event across all entries in given order.

    Entries are shuffled once per call when seed is provided, giving a
    reproducible but different ordering each epoch when the generator is
    recreated (tf.data.Dataset.from_generator recreates it each epoch).
    """
    import random
    from pid_lib.data_io import load_aligned_file, mark_no_hit_features
    from pid_lib.normalize import apply_norm

    shuffled = list(entries)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(shuffled)

    def _gen():
        for entry in shuffled:
            x, _, _ = load_aligned_file(entry)
            x, _ = mark_no_hit_features(x)
            x = apply_norm(x, norm_stats, fill_nan=fill_nan)
            label = entry.label
            for i in range(x.shape[0]):
                yield x[i], label

    return _gen


def build_dataset(tf, entries, norm_stats, fill_nan, batch_size,
                  shuffle=False, seed=None, shuffle_buffer=8192):
    """Single flat dataset from a list of FileEntry objects.

    For training: pass shuffle=True so a large tf.data buffer mixes classes.
    For val/test:  shuffle=False for deterministic, reproducible evaluation.
    """
    from pid_lib.config import FEATURE_NAMES, N_PMT

    feature_num = len(FEATURE_NAMES)
    out_sig = (
        tf.TensorSpec(shape=(N_PMT, feature_num), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    # Pass a fixed seed so each call to from_generator (i.e. each epoch)
    # uses the SAME shuffled entry order → stable gradients across epochs.
    # (The in-pipeline shuffle below adds per-epoch randomness.)
    gen_fn = make_flat_generator(entries, norm_stats, fill_nan, seed=seed)
    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=out_sig)

    if shuffle:
        # Large buffer to mix all three classes throughout each epoch.
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed,
                        reshuffle_each_iteration=True)

    opts = tf.data.Options()
    opts.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA)
    ds = ds.with_options(opts)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── DeepSphere model ─────────────────────────────────────────────────────────
def build_model(tf, feature_num, nside, indices, which_pixel_tensor,
                count_pix_const):
    from tensorflow.keras.layers import Input, Lambda, Dense
    from tensorflow.keras.models import Model
    from deepsphere import HealpyGCNN
    from deepsphere import healpy_layers as hp_layer

    npix = count_pix_const.shape[1]

    group_input = Input(shape=(17612, feature_num), name="elec_features")

    def pmt_to_pix(inp):
        b = tf.shape(inp)[0]
        pix = tf.zeros((b, npix, feature_num))
        bi = tf.range(b)[:, tf.newaxis]
        bi = tf.tile(bi, [1, tf.shape(which_pixel_tensor)[0]])
        idx = tf.stack([bi, tf.tile(tf.expand_dims(which_pixel_tensor, 0),
                                    [b, 1])], axis=-1)
        pix = tf.tensor_scatter_nd_add(pix, idx, inp)
        return pix / count_pix_const

    pixel_arrays = Lambda(pmt_to_pix)(group_input)

    layers = [
        hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=48, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=24, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=12, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyChebyshev(K=10, Fout=6, use_bias=True, use_bn=True,
                                 activation="relu"),
        hp_layer.HealpyPool(p=1),
        hp_layer.HealpyChebyshev(K=10, Fout=3),
        tf.keras.layers.Flatten(),
        Dense(3),
    ]
    ds_model = HealpyGCNN(nside=nside, indices=indices, layers=layers,
                          n_neighbors=40)
    ds_model.build(input_shape=(None, len(indices), feature_num))
    logits = ds_model(pixel_arrays)
    return Model(inputs=group_input, outputs=logits)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    tf = setup_tf(args.gpu)

    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    from pid_lib.config import CLASS_LABEL, FEATURE_NAMES, NSIDE, WHICH_PIXEL_PATH
    from pid_lib.data_io import load_aligned_file
    from pid_lib.metrics import evaluate_predictions, save_class_mapping
    from pid_lib.normalize import estimate_norm_stats, load_norm_stats, save_norm_stats
    from pid_lib.splits import build_or_load_splits

    os.makedirs(args.output_dir, exist_ok=True)
    save_class_mapping(args.output_dir)

    max_files = 3 if args.smoke_test else args.max_files_per_class
    train_e, val_e, test_e = build_or_load_splits(
        args.output_dir, seed=args.seed,
        max_files_per_class=max_files,
        reuse=not args.smoke_test,
    )
    print(f"Split (files): train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    # ── Normalization stats (training data only) ─────────────────────────────
    norm_path = os.path.join(args.output_dir, "norm_stats.json")
    if os.path.isfile(norm_path):
        norm_stats = load_norm_stats(norm_path)
        print("[norm] Loaded existing norm_stats.json")
    else:
        norm_stats = estimate_norm_stats(train_e)
        save_norm_stats(args.output_dir, norm_stats)

    # ── Class counts (no separate class_weight: equal-weight sampling already
    # ── ensures balanced batches, adding class_weight would over-correct) ────
    from collections import Counter
    label_counts = Counter()
    for e in train_e:
        _, labels, meta = load_aligned_file(e)
        label_counts[e.label] += meta.get("n_kept", len(labels))
    print(f"Train events per class: {dict(label_counts)}")
    class_weight = None   # equal-weight interleaving handles imbalance

    # ── Separate entries by class for balanced sampling ──────────────────────
    from collections import defaultdict
    entries_by_class = defaultdict(list)
    for e in train_e:
        entries_by_class[e.label].append(e)
    val_by_class = defaultdict(list)
    for e in val_e:
        val_by_class[e.label].append(e)

    batch = args.batch_size
    # Training: globally shuffled flat stream (buffer_size=8192 mixes classes).
    train_ds = build_dataset(tf, train_e, norm_stats, args.fill_nan,
                             batch, shuffle=True, seed=args.seed,
                             shuffle_buffer=args.shuffle_buffer)
    # Validation & test: fixed, deterministic order (same every epoch).
    val_ds   = build_dataset(tf, val_e,   norm_stats, args.fill_nan,
                             batch, shuffle=False, seed=args.seed)
    test_ds  = build_dataset(tf, test_e,  norm_stats, args.fill_nan,
                             batch, shuffle=False, seed=args.seed)

    # Count test events per class for reporting
    test_label_counts = Counter()
    for e in test_e:
        _, labels, meta = load_aligned_file(e)
        test_label_counts[e.label] += meta.get("n_kept", len(labels))
    val_label_counts = Counter()
    for e in val_e:
        _, labels, meta = load_aligned_file(e)
        val_label_counts[e.label] += meta.get("n_kept", len(labels))
    print(f"Val  events per class: {dict(val_label_counts)}")
    print(f"Test events per class: {dict(test_label_counts)}")

    # ── HEALPix projection ───────────────────────────────────────────────────
    wp = np.load(WHICH_PIXEL_PATH)
    which_pixel = tf.cast(wp[:, 1], tf.int32)
    indices = np.arange(hp.nside2npix(NSIDE))
    npix = len(indices)
    count_pix = np.zeros(npix)
    for i in range(17612):
        count_pix[int(wp[i, 1])] += 1
    count_pix[count_pix == 0] = 1
    count_pix = tf.constant(count_pix.reshape(1, npix, 1).astype(np.float32))

    feature_num = len(FEATURE_NAMES)

    # ── Build and compile model ───────────────────────────────────────────────
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(tf, feature_num, NSIDE, indices,
                            which_pixel, count_pix)
        model.summary(110)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    ckpt = os.path.join(args.output_dir, "checkpoint_pid")
    if os.path.isfile(ckpt + ".index"):
        model.load_weights(ckpt)
        print("[ckpt] Loaded existing checkpoint.")

    class LrDecay(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            old = self.model.optimizer.learning_rate.read_value()
            self.model.optimizer.learning_rate.assign(old * args.lr_decay)

    class LossLogger(tf.keras.callbacks.Callback):
        def __init__(self, path):
            super().__init__()
            self.path = path
        def on_epoch_end(self, epoch, logs=None):
            with open(self.path, "a") as f:
                f.write(f"Epoch {epoch+1}: "
                        f"loss={logs.get('loss'):.4f}  "
                        f"acc={logs.get('accuracy'):.4f}  "
                        f"val_loss={logs.get('val_loss', float('nan')):.4f}  "
                        f"val_acc={logs.get('val_accuracy', float('nan')):.4f}\n")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.early_stop_patience,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt, save_weights_only=True, monitor="val_loss",
                        mode="min", save_best_only=True, verbose=1),
        LrDecay(),
        LossLogger(os.path.join(args.output_dir, "loss_log.txt")),
    ]

    epochs = args.smoke_epochs if args.smoke_test else args.epochs
    t0 = time.time()
    fit_kwargs = dict(validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight
    history = model.fit(train_ds, **fit_kwargs)
    elapsed = time.time() - t0

    # ── Learning curve ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "learning_curve.png"), dpi=200)
    plt.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    # Predict directly on the batched dataset to avoid shape issues
    y_prob_all = model.predict(test_ds, verbose=1)
    y_prob_all = tf.nn.softmax(y_prob_all).numpy()

    # Collect true labels in the same order
    y_true_all = []
    for _, yb in test_ds:
        y_true_all.append(yb.numpy())
    y_true_all = np.concatenate(y_true_all)

    metrics = evaluate_predictions(y_true_all, y_prob_all, args.output_dir)

    # ── detail.txt ────────────────────────────────────────────────────────────
    _h, _r = divmod(elapsed, 3600); _m, _s = divmod(_r, 60)
    h, min_, s = int(_h), int(_m), int(_s)
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [float("nan")]))
    with open(os.path.join(args.output_dir, "detail.txt"), "w") as f:
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"command: {' '.join(sys.argv)}\n")
        f.write(f"gpu: {args.gpu}\n")
        f.write(f"class_mapping: {json.dumps(CLASS_LABEL)}\n")
        f.write(f"feature_order: {FEATURE_NAMES}\n")
        f.write(f"fill_nan: {args.fill_nan} (no-hit PMT value after norm)\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"train_events: {dict(label_counts)}\n")
        f.write(f"val_events:   {dict(val_label_counts)}\n")
        f.write(f"test_events:  {dict(test_label_counts)}\n")
        f.write(f"batch_size: {batch}\n")
        f.write(f"learning_rate_init: {args.learning_rate}\n")
        f.write(f"lr_decay: {args.lr_decay}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")
        f.write(f"shuffle_buffer: {args.shuffle_buffer}\n")
        f.write(f"epochs_trained: {len(history.history['loss'])}\n")
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"best_val_accuracy: {best_val_acc:.6f}\n")
        f.write(f"test_accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"test_macro_auc: {metrics['macro_auc']:.6f}\n")
        for cls, cls_metric in metrics["per_class"].items():
            f.write(f"test_{cls}: P={cls_metric['precision']:.4f} R={cls_metric['recall']:.4f} "
                    f"F1={cls_metric['f1']:.4f} N={cls_metric['support']}\n")
        f.write(f"elapsed: {h}h {min_}m {s}s\n")
        f.write("\nKEY FIX vs. previous run:\n")
        f.write("  - Per-class equal-weight dataset sampling "
                "(no more sequential numu→nue→nc)\n")
        f.write(f"  - fill_nan={args.fill_nan} "
                "(distinct from normalised 0)\n")

    print(f"\n[done] Test accuracy: {metrics['accuracy']:.4f}  "
          f"Macro AUC: {metrics['macro_auc']:.4f}")
    print(json.dumps(metrics["per_class"], indent=2))


if __name__ == "__main__":
    main()
