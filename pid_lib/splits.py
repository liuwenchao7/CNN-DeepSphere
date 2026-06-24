"""File-level train/validation/test splits and manifest I/O."""


import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pid_lib.data_io import FileEntry, discover_valid_entries


def stratified_file_split(
    entries: Sequence[FileEntry],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[FileEntry], List[FileEntry], List[FileEntry]]:
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[FileEntry]] = {}
    for e in entries:
        by_class.setdefault(e.label, []).append(e)

    train, val, test = [], [], []
    for label, group in sorted(by_class.items()):
        ids = list(group)
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            # Too few files: put all in train, note in manifest metadata.
            train.extend(ids)
            continue
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val :])
    return train, val, test


def entry_to_dict(e: FileEntry) -> Dict:
    return {
        "class": e.cls_name,
        "file_id": e.file_id,
        "label": e.label,
        "root": e.root,
    }


def dict_to_entry(d: Dict) -> FileEntry:
    return FileEntry(
        cls_name=d["class"],
        file_id=int(d["file_id"]),
        label=int(d["label"]),
        root=d["root"],
    )


def save_manifests(
    output_dir: str,
    train: Sequence[FileEntry],
    val: Sequence[FileEntry],
    test: Sequence[FileEntry],
    seed: int = 42,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta = {"seed": seed, "train": len(train), "val": len(val), "test": len(test)}
    if extra:
        meta.update(extra)
    with open(os.path.join(output_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    for name, items in ("train", train), ("val", val), ("test", test):
        path = os.path.join(output_dir, f"manifest_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([entry_to_dict(e) for e in items], f, indent=2)


def load_manifest(path: str) -> List[FileEntry]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [dict_to_entry(d) for d in data]


def build_or_load_splits(
    output_dir: str,
    seed: int = 42,
    max_files_per_class: Optional[int] = None,
    reuse: bool = True,
) -> Tuple[List[FileEntry], List[FileEntry], List[FileEntry]]:
    manifest_train = os.path.join(output_dir, "manifest_train.json")
    if reuse and os.path.isfile(manifest_train):
        return (
            load_manifest(manifest_train),
            load_manifest(os.path.join(output_dir, "manifest_val.json")),
            load_manifest(os.path.join(output_dir, "manifest_test.json")),
        )
    entries = discover_valid_entries()
    if max_files_per_class is not None:
        trimmed = []
        per_class = {}
        for e in entries:
            per_class.setdefault(e.cls_name, []).append(e)
        for cls_name, group in per_class.items():
            trimmed.extend(sorted(group, key=lambda x: x.file_id)[:max_files_per_class])
        entries = trimmed
    train, val, test = stratified_file_split(entries, seed=seed)
    save_manifests(output_dir, train, val, test, seed=seed)
    return train, val, test
