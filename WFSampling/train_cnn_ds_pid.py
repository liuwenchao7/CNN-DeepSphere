#!/usr/bin/env python3
"""Compatibility wrapper for legacy launch path."""

import os
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "train_cnn_ds_pid.py")

if __name__ == "__main__":
    runpy.run_path(TARGET, run_name="__main__")
