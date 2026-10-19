"""MIT-BIH Arrhythmia Database Benchmarking Script for PhysioLab Intelligence.
Evaluates ECG R-peak detection accuracy against expert cardiologist annotations using wfdb."""
import numpy as np
import wfdb
from physiolab.engines import ecg

def evaluate_record(record_name='100', database='mitdb'):
    print(f"Fetching record {record_name} from PhysioNet ({database})...")
    
    # Read record signal and expert annotations (.atr) directly from PhysioNet servers
    record = wfdb.rdrecord(record_name, pn_dir=database)
    ann = wfdb.rdann(record_name, 'atr', pn_dir=database)
    
    # Extract lead MLII (typically channel 0) and sampling frequency
    signal = record.p_signal[:, 0]
    fs = record.fs
    
    print("Running PhysioLab ECG analysis engine...")
    _, detected_peaks = ecg.analyse(signal, fs, f"MIT-BIH {record_name}")
    
    # Ground truth expert peak sample indices
    true_peaks = np.array(ann.sample)
    
    # Clinical matching tolerance window: 150 ms (ANSI/AAMI standard)
    tolerance = int(0.15 * fs)
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    
    matched_true = set()
    for p in detected_peaks:
        diffs = np.abs(true_peaks - p)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance and min_idx not in matched_true:
            tp += 1
            matched_true.add(min_idx)
        else:
            fp += 1
            
    fn = len(true_peaks) - len(matched_true)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n==========================================")
    print(f"  BENCHMARK RESULTS: MIT-BIH RECORD {record_name}")
    print(f"==========================================")
    print(f"  Expert Annotated Beats: {len(true_peaks)}")
    print(f"  Algorithmic Detections: {len(detected_peaks)}")
    print(f"  True Positives (TP):    {tp}")
    print(f"  False Positives (FP):   {fp}")
    print(f"  False Negatives (FN):   {fn}")
    print(f"------------------------------------------")
    print(f"  Sensitivity (Recall):   {sensitivity * 100:.2f}%")
    print(f"  Precision (PPV):        {ppv * 100:.2f}%")
    print(f"==========================================")
    
    return {"sensitivity": sensitivity, "ppv": ppv, "tp": tp, "fp": fp, "fn": fn}

if __name__ == "__main__":
    # Run benchmark on standard record 100
    evaluate_record('100')