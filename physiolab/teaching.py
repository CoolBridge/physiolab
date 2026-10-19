"""Teaching helpers: parameterised ECG, EEG, and Echocardiography simulators and guided-exercise logic.
All data are synthetic, so no patient data is involved."""
import numpy as np
from scipy.signal import welch, butter, lfilter


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


def simulate_eeg_pipeline(state="Eyes closed", duration_sec=10, sfreq=250, alpha_amp_mult=1.0, noise_level=0.05):
    """Generates synthetic multi-channel EEG and computes multi-stage pipeline metrics for teaching."""
    n_samples = int(duration_sec * sfreq)
    t = np.arange(n_samples) / sfreq
    ch_names = ["Fp1", "C3", "O1", "T3"]
    data = np.zeros((len(ch_names), n_samples))
    
    rng = np.random.default_rng(123)
    for i, ch in enumerate(ch_names):
        # Base background noise + artifact noise
        noise = rng.normal(0, 3.0 + (noise_level * 30), n_samples)
        
        # Alpha rhythm (10 Hz) prominent in posterior channel (O1) during eyes closed
        is_occipital = (ch == "O1")
        base_alpha = 22.0 if (state == "Eyes closed" and is_occipital) else (4.0 if state == "Eyes closed" else 1.0)
        alpha_wave = (base_alpha * alpha_amp_mult) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, np.pi))
        
        # Slow delta/theta background
        slow_wave = 7.0 * np.sin(2 * np.pi * 2.2 * t)
        
        data[i] = noise + alpha_wave + slow_wave
        
    # Pipeline stages simulation
    # Stage 2: Alpha Band-pass filter (8-13 Hz)
    nyq = sfreq / 2.0
    b, a = butter(4, [8.0 / nyq, 13.0 / nyq], btype='band')
    filtered_data = np.array([lfilter(b, a, ch) for ch in data])
    
    # Stage 3: Squared signal energy envelope
    energy_data = filtered_data ** 2
    
    # Welch PSD for global band power
    freqs, psds = welch(data, fs=sfreq, nperseg=min(256, n_samples))
    
    return {
        "t": t,
        "raw": data,
        "filtered": filtered_data,
        "energy": energy_data,
        "ch_names": ch_names,
        "freqs": freqs,
        "psds": psds
    }


def simulate_echo_pipeline(lvid_d=50.0, lvid_s=30.0, heart_rate=72.0):
    """Simulates multi-stage echocardiography mechanics, caliper geometry, and derived hemodynamics."""
    fs = ((lvid_d - lvid_s) / lvid_d) * 100.0 if lvid_d > 0 else 0.0
    
    # Volumetric approximations (Teichholz / simplified geometry)
    edv = 7.0 / (2.4 + lvid_d) * (lvid_d ** 3)
    esv = 7.0 / (2.4 + lvid_s) * (lvid_s ** 3)
    stroke_volume = max(edv - esv, 0.0)
    ejection_fraction = (stroke_volume / max(edv, 1.0)) * 100.0
    cardiac_output = (stroke_volume * heart_rate) / 1000.0
    
    # Synthetic 2D frame grid representation
    h, w = 180, 240
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    
    rx_d = int(lvid_d * 1.6)
    ry_d = int(lvid_d * 1.1)
    mask_d = ((x - cx)**2 / max(rx_d**2, 1) + (y - cy)**2 / max(ry_d**2, 1)) <= 1.0
    
    frame = np.ones((h, w), dtype=float) * 35.0
    frame[mask_d] = 190.0
    
    status = "Normal systolic function"
    if fs < 25.0:
        status = "Reduced systolic function (Possible impairment)"
    elif fs > 45.0:
        status = "Hyperdynamic function"
        
    return {
        "frame": frame,
        "lvid_d": lvid_d,
        "lvid_s": lvid_s,
        "fractional_shortening_pct": round(fs, 1),
        "stroke_volume_ml": round(stroke_volume, 1),
        "ejection_fraction_pct": round(ejection_fraction, 1),
        "cardiac_output_lmin": round(cardiac_output, 2),
        "clinical_status": status,
        "calipers": {
            "d_line": ((cx - rx_d, cy), (cx + rx_d, cy)),
            "s_line": ((cx - int(lvid_s * 1.6), cy), (cx + int(lvid_s * 1.6), cy))
        }
    }


EXERCISES = {
    "Exercise 1: Estimate heart rate by hand": {
        "aim": "Learn how heart rate is derived from R-R intervals, then compare your estimate with the algorithm.",
        "steps": ["Count the QRS complexes in the 10-second window.", "Multiply by 6 to get beats per minute.",
                  "Enter your estimate, then reveal the algorithm's result."],
    },
    "Exercise 2: Noise and signal quality": {
        "aim": "See how baseline wander and mains interference degrade detection and quality scoring.",
        "steps": ["Start with no noise and note the heart rate and quality label.",
                  "Increase mains interference and baseline wander.",
                  "Record when detection accuracy drops and the quality label changes."],
    },
    "Exercise 3: Heart rate variability": {
        "aim": "Relate beat-to-beat variability to SDNN and RMSSD.",
        "steps": ["Set variability to 0, then 3, then 8 bpm.", "Record SDNN and RMSSD at each setting.",
                  "Explain why higher variability raises both values."],
    },
}