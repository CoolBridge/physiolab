"""EEG engine: load EDF/CSV, channel QC, band power, alpha peak."""
import os, tempfile
import numpy as np
import pandas as pd
from scipy.signal import welch
from ..schema import AnalysisResult, make_provenance

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_edf(file_bytes: bytes):
    """Return (data_uV [ch x n], fs, channel_names)."""
    import mne
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        f.write(file_bytes)
        path = f.name
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        return raw.get_data() * 1e6, float(raw.info["sfreq"]), list(raw.ch_names)
    finally:
        os.unlink(path)


def load_csv(file, fs=None):
    df = pd.read_csv(file)
    num = df.select_dtypes("number")
    tcols = [c for c in num.columns if c.lower() in {"time", "t", "seconds", "time_s", "sec"}]
    chans = [c for c in num.columns if c not in tcols]
    if tcols:
        fs = 1.0 / np.median(np.diff(num[tcols[0]].to_numpy(float)))
    if not fs:
        raise ValueError("Sampling rate unknown: add a time column or enter it manually.")
    return num[chans].to_numpy(float).T, float(fs), chans


def analyse(data, fs, names, source_name="eeg", line_hz=50):
    nper = int(min(data.shape[1], 4 * fs))
    per_ch, bad, flags, alpha_peaks = {}, [], [], []
    stds = np.nanstd(data, axis=1)
    med = np.median(stds[stds > 0]) if np.any(stds > 0) else 1.0
    for i, ch in enumerate(names):
        x = np.nan_to_num(data[i] - np.nanmean(data[i]))
        f, p = welch(x, fs, nperseg=nper)
        tot = _trapz(p[(f >= 0.5) & (f <= 45)], f[(f >= 0.5) & (f <= 45)]) + 1e-20
        rel = {b: round(float(_trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / tot), 3)
               for b, (lo, hi) in BANDS.items()}
        ln = (f > line_hz - 1) & (f < line_hz + 1)
        line_ratio = float(_trapz(p[ln], f[ln]) / (_trapz(p[f > 1], f[f > 1]) + 1e-20)) if ln.any() else 0
        status = "ok"
        if stds[i] < 1e-3: status = "flat"
        elif stds[i] > 5 * med: status = "noisy"
        elif line_ratio > 0.3: status = "mains-noise"
        if status != "ok": bad.append(ch)
        a = (f >= 8) & (f <= 13)
        if status == "ok" and a.any(): alpha_peaks.append(float(f[a][np.argmax(p[a])]))
        per_ch[ch] = {"std_uV": round(float(stds[i]), 2), "status": status,
                      **{f"rel_{k}": v for k, v in rel.items()}}
    good = [c for c in names if c not in bad]
    glob = {b: round(float(np.mean([per_ch[c][f"rel_{b}"] for c in good])), 3) for b in BANDS} if good else {}
    q = {"channels": len(names), "flagged_channels": bad, "good_channel_fraction": round(len(good) / len(names), 2),
         "duration_s": round(data.shape[1] / fs, 1), "sampling_rate_hz": round(fs, 1)}
    m = {"global_relative_power": glob, "per_channel": per_ch}
    if alpha_peaks: m["alpha_peak_hz"] = round(float(np.median(alpha_peaks)), 2)
    if bad: flags.append(f"Channels flagged: {', '.join(bad)}.")
    if glob and glob["delta"] > 0.6: flags.append("Delta-dominant spectrum: check for artefact or low-frequency drift.")
    if q["good_channel_fraction"] < 0.5: flags.append("Fewer than half of channels usable.")
    limits = ["Spectral summary only: no seizure, spike or sleep-stage detection.",
              "CSV input is assumed to be in microvolts."]
    return AnalysisResult("EEG", source_name, q, m, flags, limits, make_provenance("Welch PSD band power + channel QC"))