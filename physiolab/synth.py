"""Synthetic demo data generators (no patient data)."""
import io
import numpy as np
import pandas as pd


def synth_ecg_csv(path, fs=250, duration=30, hr=72, seed=1):
    import neurokit2 as nk
    sig = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr, noise=0.03, random_state=seed)
    pd.DataFrame({"time_s": np.arange(len(sig)) / fs, "lead_II_mV": sig}).to_csv(path, index=False)


def synth_eeg_csv(path, fs=250, duration=20, seed=2, line_hz=50):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration)) / fs
    names = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Cz", "Pz"]
    cols = {"time_s": t}
    for n in names:
        alpha_amp = 25 if n in ("O1", "O2", "Pz") else 8
        x = (alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6)) + 6 * np.sin(2 * np.pi * 6 * t)
             + 4 * np.sin(2 * np.pi * 20 * t) + rng.normal(0, 5, t.size))
        cols[n] = x
    cols["Fp2"] = cols["Fp2"] * 0 + 0.0  # simulate a disconnected (flat) electrode
    cols["T7"] = 5 * rng.normal(0, 40, t.size) + 120 * np.sin(2 * np.pi * line_hz * t)  # noisy + mains
    pd.DataFrame(cols).to_csv(path, index=False)


def synth_echo_dicom(path, size=256, frames=10, mm_per_px=0.3):
    """Beating-ventricle phantom as a multi-frame ultrasound DICOM with calibration."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(3)
    arr = []
    for i in range(frames):
        r = 40 + 14 * np.sin(2 * np.pi * i / frames)
        img = 90 + rng.normal(0, 12, (size, size))
        img[((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < r ** 2] = 15
        arr.append(np.clip(img, 0, 255).astype(np.uint8))
    arr = np.stack(arr)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.PatientID = "US", "SYNTHETIC^DEMO", "SYN001"
    ds.Rows, ds.Columns, ds.NumberOfFrames = size, size, frames
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation, ds.FrameTime = 7, 0, "33.3"
    reg = Dataset()
    reg.PhysicalUnitsXDirection = 3
    reg.PhysicalUnitsYDirection = 3
    reg.PhysicalDeltaX = mm_per_px / 10
    reg.PhysicalDeltaY = mm_per_px / 10
    ds.SequenceOfUltrasoundRegions = [reg]
    ds.PixelData = arr.tobytes()
    ds.save_as(path, enforce_file_format=True)


def make_all(folder="demo_data"):
    import os
    os.makedirs(folder, exist_ok=True)
    synth_ecg_csv(f"{folder}/demo_ecg.csv")
    synth_eeg_csv(f"{folder}/demo_eeg.csv")
    synth_echo_dicom(f"{folder}/demo_echo.dcm")


if __name__ == "__main__":
    make_all()