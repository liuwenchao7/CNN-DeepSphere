"""Data roots, class mapping, and feature definitions."""


# AGENTS.md lists /disk_pool1/liuxy/nue/npy but that path is empty on disk;
# actual nue data lives under nu_e/npy (documented in audit output).
CLASS_ROOTS = {
    "numu": "/disk_pool1/weijsh/waveform/npy",
    "nue": "/disk_pool1/liuxy/nu_e/npy",
    "nc": "/disk_pool1/liuxk/Muon/J24/nc/npy",
}

CLASS_LABEL = {"numu": 0, "nue": 1, "nc": 2}
LABEL_TO_CLASS = {v: k for k, v in CLASS_LABEL.items()}

N_PMT = 17612
NSIDE = 32
WHICH_PIXEL_PATH = "/disk_pool1/liuwc/ds_cnn/whichPixel_nside32_LCDpmts.npy"
PMT_TYPE_CSV = "/disk_pool1/liuwc/data/cnn+ds/PMTType_CD_LPMT.csv"

FEATURE_NAMES = ["fht", "npe", "nperatio4", "peak", "peaktime", "slope4"]
FEATURE_FILE_PREFIX = {
    "fht": "x_fht_pmt",
    "npe": "x_npe_pmt",
    "nperatio4": "x_nperatio4_pmt",
    "peak": "x_peak_pmt",
    "peaktime": "x_peaktime_pmt",
    "slope4": "x_slope4_pmt",
}

SUBDIRS = ("det_fea", "elec_fea", "waveform", "y")

# FHT / peaktime no-hit sentinels observed in data audit.
FHT_NO_HIT_MAX = 0          # nue uses negative FHT for no-hit
PEAKTIME_NO_HIT_MIN = 1008  # time window end sentinel

WFSAMPLING_OUTPUT_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling"
WFSAMPLING_DECON_WAVEFORM_ROOT = "/disk_pool1/liuwc/data/cnn+ds/pid/WFSampling_decon_waveform"
