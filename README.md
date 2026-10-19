# PhysioLab Intelligence (V0.1)

Modular physiological data platform. V1 modalities: ECG, EEG, Echocardiography.
Research and education prototype. Not for diagnosis.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m physiolab.synth        # regenerates synthetic demo files
pytest -q                        # engine tests
ADMIN_PASSWORD=choose-one streamlit run app.py
```

## Layout
- `physiolab/schema.py`    shared result schema + provenance
- `physiolab/engines/`     ecg.py, eeg.py, echo.py (no UI code)
- `physiolab/explain.py`   audience-specific explanations from structured results only
- `physiolab/db.py`        registration + usage tracking (SQLite; swap for Supabase/PostgreSQL)
- `app.py`                 Streamlit interface
- `tests/`                 pytest suite

## Teaching lab (V0.2)
- Simulator with five parameters (rate, variability, noise, baseline wander, 50 Hz interference).
- "Show the steps" view: raw, filtered, energy and threshold, detected R-peaks.
- Three guided exercises (hand-counted heart rate with answer check, noise and quality, HRV).
- ECG quality scoring now detects baseline wander and mains interference.

## Rules
1. A format is advertised only after its pathway is tested.
2. The explanation layer never sees raw signals and never invents numbers.
3. Use de-identified or synthetic data only. Do not store patient files.

## Next
Replace synthetic data with PhysioNet ECG and MNE/CHB-MIT EEG validation sets, add email OTP (Supabase Auth),
click-to-place calipers (streamlit-image-coordinates), FastAPI service layer, local LLM via Ollama.