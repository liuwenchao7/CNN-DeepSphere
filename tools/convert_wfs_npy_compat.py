#!/usr/bin/env python3
"""Convert WFS v2 .npy (numpy2 pickle) to deepsphere-compatible list-based .npy."""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

READER = "/disk_pool1/liuwc/anaconda3/envs/cnn_deepsphere/bin/python3"
READER_ENV = "/disk_pool1/liuwc/anaconda3/envs/cnn_deepsphere/lib"
WRITER = "/disk_pool1/liuwc/anaconda3/envs/deepsphere/bin/python3"


def _is_compat(path):
    try:
        obj = np.load(path, allow_pickle=True).item()
        return isinstance(obj["time"][0], list)
    except ModuleNotFoundError:
        return False
    except Exception:
        return False


def _convert_with_reader(path):
    """Read with numpy2 env, save list payload with deepsphere numpy."""
    if _is_compat(path):
        return "skip", path
    import subprocess
    import tempfile

    read_script = f"""
import numpy as np, pickle, sys
path = {path!r}
obj = np.load(path, allow_pickle=True).item()
compat = {{
    "time": [np.asarray(t, dtype=np.float32).tolist() for t in obj["time"]],
    "adc": [np.asarray(a, dtype=np.float32).tolist() for a in obj["adc"]],
    "offsets": [np.asarray(o, dtype=np.int64).tolist() for o in obj["offsets"]],
    "n_events": int(obj["n_events"]),
    "file_id": int(obj["file_id"]),
    "n_pmt": int(obj.get("n_pmt", 17612)),
    "empty_frac": float(obj.get("empty_frac", 0.0)),
    "wfs_version": int(obj.get("wfs_version", 2)),
}}
pickle.dump(compat, sys.stdout.buffer, protocol=2)
"""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = READER_ENV + ":" + env.get("LD_LIBRARY_PATH", "")
    r1 = subprocess.run(
        [READER, "-c", read_script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if r1.returncode != 0:
        return "error", f"{path}: read failed: {r1.stderr.decode()}"

    tmp_pkl = path + ".tmplists.pkl"
    with open(tmp_pkl, "wb") as f:
        f.write(r1.stdout)

    write_script = f"""
import numpy as np, pickle, os
path = {path!r}
pkl = path + ".tmplists.pkl"
with open(pkl, "rb") as f:
    c = pickle.load(f)
payload = {{
    "time": c["time"],
    "adc": c["adc"],
    "offsets": c["offsets"],
    "n_events": c["n_events"],
    "file_id": c["file_id"],
    "n_pmt": c["n_pmt"],
    "empty_frac": c["empty_frac"],
    "wfs_version": c["wfs_version"],
}}
tmp = path + ".tmpcompat.npy"
np.save(tmp, payload, allow_pickle=True)
os.replace(tmp, path)
os.remove(pkl)
print("ok")
"""
    r2 = subprocess.run([WRITER, "-c", write_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r2.returncode != 0:
        if os.path.isfile(tmp_pkl):
            os.remove(tmp_pkl)
        return "error", f"{path}: write failed: {r2.stderr.decode()}"
    return "ok", path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(args.root, "*", "wfsampling_*.npy"))
        if ".tmp" not in os.path.basename(f)
    )
    print(f"Found {len(files)} files under {args.root}")

    results = {}
    worker = _convert_with_reader
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(worker, f): f for f in files}
        for fut in as_completed(futs):
            status, msg = fut.result()
            results[status] = results.get(status, 0) + 1
            if status == "error":
                print(f"  [error] {msg}")

    print("Summary:", results)


if __name__ == "__main__":
    main()
