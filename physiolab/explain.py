"""Explanation engine. Reads ONLY the structured result; never raw signals.
Deterministic templates guarantee numbers are never invented. An LLM can be
plugged in via `llm_prompt()`, with the structured JSON as its sole input."""
import json

AUDIENCES = ["Student", "Lecturer", "Researcher", "Practitioner"]


def _ecg(m, q, a):
    hr, sd = m.get("heart_rate_bpm"), m.get("sdnn_ms")
    if hr is None:
        return "Not enough clean beats were found to report heart rate or variability."
    base = f"{m['beats_detected']} beats detected, mean heart rate {hr} bpm, signal quality {q['label']}."
    if a == "Student":
        return (base + f" Heart rate is 60,000 divided by the mean R-R interval ({m['mean_rr_ms']} ms). "
                f"SDNN ({sd} ms) and RMSSD ({m['rmssd_ms']} ms) describe beat-to-beat variability, "
                "which reflects autonomic (vagal) influence on the sinoatrial node.")
    if a == "Lecturer":
        return (base + " Teaching points: how R-peak detection works (band-pass, energy, thresholding), "
                "why signal quality gates every downstream metric, and how time-domain HRV links to autonomic tone.")
    if a == "Researcher":
        iv = ", ".join(f"{k}={m[k]}" for k in ("pr_ms", "qrs_ms", "qt_ms") if k in m) or "intervals unavailable"
        return (base + f" HRV: SDNN {sd}, RMSSD {m['rmssd_ms']}, pNN50 {m['pnn50_pct']}%, CV {m['rr_variability_cv']}. "
                f"Intervals: {iv}. Export JSON for downstream statistics; see provenance for method versions.")
    return (base + " Review flagged observations against the raw trace before any clinical use; "
            "this output is decision-support context, not a diagnosis.")


def _eeg(m, q, a):
    g = m.get("global_relative_power", {})
    if not g:
        return "No usable channels were available for spectral analysis."
    dom = max(g, key=g.get)
    base = (f"{q['channels']} channels, {q['good_channel_fraction']*100:.0f}% usable. "
            f"Dominant band: {dom} ({g[dom]*100:.0f}% of 0.5-45 Hz power).")
    ap = m.get("alpha_peak_hz")
    if a == "Student":
        return base + (f" Alpha peak near {ap} Hz. Alpha (8-13 Hz) is typically strongest over posterior regions "
                       "in a relaxed, eyes-closed state." if ap else "")
    if a == "Lecturer":
        return base + " Teaching points: Welch spectral estimation, relative vs absolute power, and why mains noise and bad channels must be handled first."
    if a == "Researcher":
        return base + " Relative band powers per channel are in the export. Consider re-referencing, ICA artefact removal and epoching before group statistics."
    return base + " Spectral summary only. Epileptiform activity, focal slowing and sleep staging require expert visual review."


def _echo(m, q, a):
    cal = m.get("caliper_mm")
    base = f"{q['frames']} frame(s); calibration {'available' if q['calibrated'] else 'missing'}."
    if cal:
        base += " Measurements: " + ", ".join(f"{k} {v} mm" for k, v in cal.items()) + "."
    if "fractional_shortening_pct" in m:
        base += f" Fractional shortening {m['fractional_shortening_pct']}%."
    tail = {"Student": " Fractional shortening = (LVIDd - LVIDs) / LVIDd, a simple index of LV systolic function.",
            "Lecturer": " Teaching points: calibration from DICOM ultrasound regions, caliper placement technique, and limits of linear measurements.",
            "Researcher": " Metadata and calipers are exported as JSON; volumetric or strain analysis would require validated segmentation.",
            "Practitioner": " Manual measurements only. Reference ranges and interpretation remain with the reporting clinician."}
    return base + tail[a]


def explain(result: dict, audience: str) -> str:
    fn = {"ECG": _ecg, "EEG": _eeg, "Echocardiography": _echo}[result["modality"]]
    return fn(result["measurements"], result["quality"], audience)


def llm_prompt(result: dict, audience: str) -> str:
    """Prompt for an optional LLM explainer (cloud now, local Ollama later)."""
    return (f"You explain physiological analysis results to a {audience}. Use ONLY the JSON below. "
            "Do not add, change or infer any numeric value. Do not diagnose. State limitations.\n"
            + json.dumps(result, indent=2))