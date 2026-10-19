"""Echo engine: DICOM/image loading, de-identified metadata, calibrated caliper measurements."""
import io
import numpy as np
from ..schema import AnalysisResult, make_provenance

SAFE_TAGS = ["Modality", "Manufacturer", "Rows", "Columns", "NumberOfFrames", "FrameTime",
             "CineRate", "HeartRate", "TransducerFrequency", "PhotometricInterpretation"]


def load_dicom(file_bytes: bytes):
    """Return (frames [n,h,w], metadata, mm_per_px or None). Patient identifiers are never read out."""
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array
    nf = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if nf == 1:
        arr = arr[None]
    if arr.ndim == 4:
        arr = arr.mean(axis=-1)
    meta = {t: str(getattr(ds, t)) for t in SAFE_TAGS if hasattr(ds, t)}
    mm = None
    try:
        r = ds.SequenceOfUltrasoundRegions[0]
        if int(r.PhysicalUnitsXDirection) == 3:  # 3 = cm
            mm = float(r.PhysicalDeltaX) * 10
    except Exception:
        pass
    if mm is None and hasattr(ds, "PixelSpacing"):
        mm = float(ds.PixelSpacing[0])
    return arr, meta, mm


def load_image(file_bytes: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(file_bytes)).convert("L")
    return np.asarray(im)[None], {"Modality": "image", "Rows": str(im.height), "Columns": str(im.width)}, None


def distance_mm(p1, p2, mm_per_px):
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * mm_per_px)


def fractional_shortening(lvidd_mm, lvids_mm):
    return round((lvidd_mm - lvids_mm) / lvidd_mm * 100, 1)


def ejection_fraction(edv_ml, esv_ml):
    return round((edv_ml - esv_ml) / edv_ml * 100, 1)


def analyse(frames, meta, mm_per_px, calipers=None, source_name="echo"):
    f = frames[0].astype(float)
    contrast = float(np.percentile(f, 95) - np.percentile(f, 5)) / (f.max() + 1e-9)
    q = {"frames": int(frames.shape[0]), "contrast": round(contrast, 2),
         "calibrated": mm_per_px is not None, "label": "good" if contrast > 0.25 else "low-contrast"}
    m = {"metadata": meta, "mm_per_pixel": round(mm_per_px, 4) if mm_per_px else None}
    flags, vals = [], {}
    if not mm_per_px:
        flags.append("No physical calibration found: enter mm per pixel to get real-world measurements.")
    for c in calipers or []:
        if mm_per_px:
            vals[c["label"]] = round(distance_mm(c["p1"], c["p2"], mm_per_px), 1)
    if vals:
        m["caliper_mm"] = vals
        if "LVIDd" in vals and "LVIDs" in vals:
            m["fractional_shortening_pct"] = fractional_shortening(vals["LVIDd"], vals["LVIDs"])
    limits = ["Manual measurement-assisted viewer. No automated chamber segmentation or diagnosis.",
              "Reference ranges are not applied; interpretation requires a clinician."]
    return AnalysisResult("Echocardiography", source_name, q, m, flags, limits,
                          make_provenance("DICOM metadata parsing + manual calipers"))