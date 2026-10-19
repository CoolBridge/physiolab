"""Teaching helpers: synthetic ECG, EEG and echo simulators plus guided-exercise logic.
All data are synthetic, so no patient data is involved."""
import numpy as np
from scipy.signal import welch

SPECIALTIES = ["ECG", "EEG", "Echocardiography"]


# ------------------------------------------------------------------ ECG
def simulate_ecg(hr=72, duration=30, fs=250, noise=0.03, variability=1.0,
                 wander=0.0, mains=0.0, seed=1):
    """Return (time, signal, truth dict). variability = heart-rate std (bpm);
    wander = baseline wander amplitude (mV); mains = 50 Hz interference amplitude (mV)."""
    import neurokit2 as nk
    sig = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr, heart_rate_std=variability,
                          noise=noise, random_state=seed)
    t = np.arange(len(sig)) / fs
    sig = sig + wander * np.sin(2 * np.pi * 0.25 * t) + mains * np.sin(2 * np.pi * 50 * t)
    return t, sig, {"heart_rate_bpm": hr, "variability_bpm": variability, "noise": noise,
                    "wander_mV": wander, "mains_mV": mains, "fs": fs}


def check_manual_hr(student_hr, computed_hr, tolerance=5):
    err = round(float(student_hr) - float(computed_hr), 1)
    return {"error_bpm": err, "within_tolerance": abs(err) <= tolerance,
            "message": ("Good estimate." if abs(err) <= tolerance else
                        "Recount the beats in the window and multiply by 6 (10-second window).")}


def check_within(student, truth, tolerance):
    """Generic numeric check. Returns error, whether within tolerance."""
    err = round(float(student) - float(truth), 2)
    return {"error": err, "within_tolerance": abs(err) <= tolerance}


# ------------------------------------------------------------------ EEG
EEG_NAMES = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Cz", "Pz"]
EEG_FREQS = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 38.0}
EEG_POSTERIOR = {"O1", "O2", "Pz"}


def simulate_eeg(amps=None, eyes="closed", noise=5.0, mains=0.0, flat_channel=False,
                 alpha_posterior_only=True, fs=250, duration=20, seed=2):
    """Return (time, data [ch x n] in uV, channel names, truth).
    amps: per-band sinusoid amplitude in uV. With alpha_posterior_only, alpha is strongest at O1/O2/Pz,
    and eyes='open' attenuates it (alpha blocking)."""
    base = {"delta": 3.0, "theta": 6.0, "alpha": 25.0, "beta": 4.0, "gamma": 0.0}
    base.update(amps or {})
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    data = []
    for ch in EEG_NAMES:
        x = rng.normal(0, noise, t.size) + mains * np.sin(2 * np.pi * 50 * t)
        for band, f in EEG_FREQS.items():
            a = base[band]
            if band == "alpha":
                if alpha_posterior_only and ch not in EEG_POSTERIOR:
                    a *= 0.35
                if eyes == "open":
                    a *= 0.25
            x = x + a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        data.append(x)
    data = np.array(data)
    if flat_channel:
        data[EEG_NAMES.index("Fp2")] = 0.0
    truth = {"amps": base, "eyes": eyes, "noise_uV": noise, "mains_uV": mains, "flat_channel": flat_channel}
    return t, data, EEG_NAMES, truth


def eeg_spectrum(data, fs, rows=None):
    """Mean Welch power spectrum across the chosen channel rows (0.5 to 45 Hz)."""
    rows = list(range(data.shape[0])) if rows is None else rows
    ps = []
    for i in rows:
        f, p = welch(data[i] - np.mean(data[i]), fs, nperseg=int(min(data.shape[1], 4 * fs)))
        ps.append(p)
    p = np.mean(ps, axis=0) if ps else np.zeros_like(f)
    sel = (f >= 0.5) & (f <= 45)
    return f[sel], p[sel]


# ------------------------------------------------------------------ Echo
def simulate_echo(lvidd_mm=48.0, lvids_mm=32.0, mm_per_px=0.4, size=300, wall_mm=10.0, seed=3):
    """Return ([diastolic_frame, systolic_frame] uint8, truth). A schematic LV cross-section:
    dark cavity (the blood pool), bright myocardial wall, speckled background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2

    def frame(d_mm):
        a = d_mm / mm_per_px / 2            # horizontal cavity radius in pixels
        b = 1.25 * a                        # slightly elongated along the long axis
        wall = wall_mm / mm_per_px
        img = 90 + rng.normal(0, 10, (size, size))
        outer = ((xx - cx) / (a + wall)) ** 2 + ((yy - cy) / (b + wall)) ** 2 < 1
        inner = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 < 1
        img[outer] = 190 + rng.normal(0, 10, outer.sum())
        img[inner] = 20 + rng.normal(0, 6, inner.sum())
        return np.clip(img, 0, 255).astype(np.uint8)

    truth = {"lvidd_mm": lvidd_mm, "lvids_mm": lvids_mm, "mm_per_px": mm_per_px,
             "fractional_shortening_pct": echo_fs(lvidd_mm, lvids_mm), "row": size // 2}
    return [frame(lvidd_mm), frame(lvids_mm)], truth


def echo_fs(lvidd_mm, lvids_mm):
    return round((lvidd_mm - lvids_mm) / lvidd_mm * 100, 1)


def cavity_extent_px(frame, row=None, threshold=50):
    """Reference measurement: the contiguous dark run containing the image centre on the chosen row.
    Returns (x_start, x_end), or None if the centre is not inside the cavity."""
    row = frame.shape[0] // 2 if row is None else row
    dark = frame[row] < threshold
    mid = frame.shape[1] // 2
    if not dark[mid]:
        return None
    lo = hi = mid
    while lo > 0 and dark[lo - 1]:
        lo -= 1
    while hi < len(dark) - 1 and dark[hi + 1]:
        hi += 1
    return int(lo), int(hi)


# ------------------------------------------------------------------ Exercises
EXERCISES = {
    "ECG": {
        "Exercise 1: Estimate heart rate by hand": {
            "aim": "Learn how heart rate is derived from R-R intervals, then compare your estimate with the algorithm.",
            "steps": ["Count the QRS complexes in the 10-second window.", "Multiply by 6 to get beats per minute.",
                      "Enter your estimate, then reveal the algorithm's result."]},
        "Exercise 2: Noise and signal quality": {
            "aim": "See how baseline wander and mains interference degrade detection and quality scoring.",
            "steps": ["Start with no noise and note the heart rate and quality label.",
                      "Increase mains interference and baseline wander in the Simulator.",
                      "Record when detection accuracy drops and the quality label changes."]},
        "Exercise 3: Heart rate variability": {
            "aim": "Relate beat-to-beat variability to SDNN and RMSSD.",
            "steps": ["Set variability to 0, then 3, then 8 bpm in the Simulator.", "Record SDNN and RMSSD at each setting.",
                      "Explain why higher variability raises both values."]},
    },
    "EEG": {
        "Exercise 1: Name the dominant rhythm": {
            "aim": "Recognise which frequency band dominates a trace, then check against the spectral analysis.",
            "steps": ["Look at the 5-second trace and estimate its dominant rhythm (count peaks per second).",
                      "Choose the band you think dominates.", "Reveal the power spectrum and compare."]},
        "Exercise 2: Eyes open versus eyes closed": {
            "aim": "Demonstrate alpha blocking (the Berger effect) quantitatively.",
            "steps": ["In the Simulator, set eyes to closed and record the global alpha relative power.",
                      "Set eyes to open and record it again.",
                      "Explain the change and which electrodes show it most."]},
        "Exercise 3: Artefacts and bad channels": {
            "aim": "See how mains interference and a disconnected electrode affect the analysis.",
            "steps": ["Add a flat electrode and note which channel is flagged.",
                      "Increase the 50 Hz interference and note when channels are flagged.",
                      "Describe how each artefact changes the band-power summary."]},
    },
    "Echocardiography": {
        "Exercise 1: Measure the ventricle": {
            "aim": "Place calipers across the LV cavity at end-diastole and end-systole, then compute fractional shortening.",
            "steps": ["Set the left and right caliper positions on the diastolic frame.",
                      "Repeat on the systolic frame.", "Compare your measurements and fractional shortening with the reference."]},
        "Exercise 2: Ejection fraction from volumes": {
            "aim": "Calculate ejection fraction from end-diastolic and end-systolic volumes.",
            "steps": ["Read the two volumes.", "Apply EF = (EDV - ESV) / EDV x 100.", "Enter your answer and check it."]},
        "Exercise 3: Why calibration matters": {
            "aim": "See that identical pixel distances mean different physical sizes under different calibrations.",
            "steps": ["In the Simulator, keep LVIDd fixed and change mm per pixel.",
                      "Note how the cavity width in pixels and in millimetres changes.",
                      "Explain why an uncalibrated measurement cannot be interpreted."]},
    },
}