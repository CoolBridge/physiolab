"""ECG engine: load, quality-check, detect beats, extract rate/HRV/intervals."""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from ..schema import AnalysisResult, make_provenance

TIME_NAMES = {"time", "t", "seconds", "time_s", "sec"}


def load_csv(file, fs=None):
    """Return (signal, fs, column_name). Infers fs from a time column if present."""
    df = pd.read_csv(file)
    num = df.select_dtypes("number")
    tcols = [c for c in num.columns if c.lower() in TIME_NAMES]
    scols = [c for c in num.columns if c not in tcols]
    if not scols:
        raise ValueError("No numeric signal column found in the CSV.")
    sig = num[scols[0]].to_numpy(float)
    if tcols:
        dt = np.median(np.diff(num[tcols[0]].to_numpy(float)))
        fs = 1.0 / dt if dt > 0 else fs
    if not fs:
        raise ValueError("Sampling rate unknown: add a time column or enter it manually.")
    return sig, float(fs), scols[0]


def _bandpass(x, fs, lo, hi, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def signal_quality(sig, fs):
    n = len(sig)
    nan_frac = float(np.mean(~np.isfinite(sig)))
    s = np.nan_to_num(sig, nan=np.nanmedian(sig))
    flat = float(np.mean(np.abs(np.diff(s)) < 1e-9))
    lo, hi = np.percentile(s, [0.5, 99.5])
    clip = float(np.mean((s <= lo + 1e-12) | (s >= hi - 1e-12)))
    band = _bandpass(s, fs, 5, 30)
    snr_like = float(np.var(band) / (np.var(s - band) + 1e-12))
    f, p = welch(s - s.mean(), fs, nperseg=int(min(n, 4 * fs)))
    tot = p[(f >= 0.1) & (f <= min(100, fs / 2 - 1))].sum() + 1e-20
    wander = float(p[(f >= 0.1) & (f < 0.7)].sum() / tot)
    mains = float(sum(p[(f >= c - 1) & (f <= c + 1)].sum() for c in (50, 60) if c < fs / 2) / tot)
    score = 1.0
    score -= min(max(wander - 0.2, 0), 0.5) + (min(mains * 1.5, 0.4) if mains > 0.04 else 0)
    score -= min(nan_frac * 2, 0.5) + min(flat, 0.5) + min(max(clip - 0.02, 0) * 5, 0.3)
    score -= 0.2 if snr_like < 0.1 else 0
    score = float(np.clip(score, 0, 1))
    label = "good" if score >= 0.75 else "fair" if score >= 0.5 else "poor"
    return {"score": round(score, 2), "label": label, "nan_fraction": round(nan_frac, 4),
            "flatline_fraction": round(flat, 4), "clipping_fraction": round(clip, 4),
            "baseline_wander_ratio": round(wander, 3), "mains_ratio": round(mains, 3),
            "duration_s": round(n / fs, 1), "sampling_rate_hz": round(fs, 1)}


def detect_stages(sig, fs):
    """Run R-peak detection and return every intermediate stage (used by teaching mode)."""
    s = np.nan_to_num(sig, nan=np.nanmedian(sig))
    filt = _bandpass(s, fs, 5, 15)
    energy = np.convolve(np.diff(filt, prepend=filt[0]) ** 2, np.ones(max(int(0.08 * fs), 1)), "same")
    thr = 0.35 * np.percentile(energy, 99)
    cand, _ = find_peaks(energy, height=thr, distance=int(0.25 * fs))
    w = int(0.06 * fs)
    peaks = []
    for c in cand:
        a, b = max(c - w, 0), min(c + w, len(s))
        peaks.append(a + int(np.argmax(np.abs(s[a:b] - np.median(s)))))
    return {"raw": s, "filtered": filt, "energy": energy, "threshold": float(thr),
            "peaks": np.unique(np.array(peaks, dtype=int))}


def detect_r_peaks(sig, fs):
    return detect_stages(sig, fs)["peaks"]


def _intervals(sig, fs):
    """Optional PR/QRS/QT via NeuroKit2 delineation; returns {} if it fails."""
    try:
        import neurokit2 as nk
        cleaned = nk.ecg_clean(sig, sampling_rate=int(fs))
        _, info = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
        _, w = nk.ecg_delineate(cleaned, info["ECG_R_Peaks"], sampling_rate=int(fs), method="dwt")

        def med(a, b):
            a, b = np.array(w[a], float), np.array(w[b], float)
            ok = ~np.isnan(a) & ~np.isnan(b)
            return float(np.median((b[ok] - a[ok]) / fs * 1000)) if ok.sum() >= 3 else None
        out = {"qrs_ms": med("ECG_R_Onsets", "ECG_R_Offsets"),
               "pr_ms": med("ECG_P_Onsets", "ECG_R_Onsets"),
               "qt_ms": med("ECG_R_Onsets", "ECG_T_Offsets")}
        return {k: round(v, 0) for k, v in out.items() if v and 20 < v < 700}
    except Exception:
        return {}


def analyse(sig, fs, source_name="ecg"):
    q = signal_quality(sig, fs)
    peaks = detect_r_peaks(sig, fs)
    m, flags = {"beats_detected": int(len(peaks))}, []
    limits = ["Single-lead analysis only; morphology is not interpreted.",
              "Interval estimates depend on signal quality and delineation accuracy."]
    if len(peaks) >= 4:
        rr = np.diff(peaks) / fs * 1000
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) >= 3:
            hr = 60000 / np.mean(rr)
            d = np.diff(rr)
            m.update({"heart_rate_bpm": round(float(hr), 1), "mean_rr_ms": round(float(rr.mean()), 0),
                      "sdnn_ms": round(float(rr.std(ddof=1)), 1),
                      "rmssd_ms": round(float(np.sqrt(np.mean(d ** 2))), 1),
                      "pnn50_pct": round(float(np.mean(np.abs(d) > 50) * 100), 1),
                      "rr_variability_cv": round(float(rr.std() / rr.mean()), 3)})
            if hr > 100: flags.append("Rate above 100 bpm in this segment.")
            if hr < 60: flags.append("Rate below 60 bpm in this segment.")
            if m["rr_variability_cv"] > 0.15: flags.append("Irregular R-R pattern: review the trace manually.")
    else:
        flags.append("Too few beats detected for rate and variability metrics.")
    if q["mains_ratio"] > 0.05: flags.append("Mains (50/60 Hz) interference detected.")
    if q["baseline_wander_ratio"] > 0.25: flags.append("Baseline wander detected.")
    if q["label"] == "poor":
        flags.append("Poor signal quality: measurements may be unreliable.")
    else:
        m.update(_intervals(sig, fs))
    return AnalysisResult("ECG", source_name, q, m, flags, limits,
                          make_provenance("bandpass energy R-peak detector + NeuroKit2 DWT delineation")), peaks