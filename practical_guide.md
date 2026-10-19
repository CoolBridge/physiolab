# PHYSIOLAB INTELLIGENCE: ACADEMIC PRACTICAL & INSTRUCTOR GUIDE

**Multimodal Computational Physiology Laboratory (ECG, EEG, Echocardiography)**
*Platform Version: V0.2 · Designed for Undergraduate & Postgraduate Physiology Curricula*

---

## PART 1: INSTRUCTOR GUIDE & LABORATORY OBJECTIVES

### 1. Overview for Course Coordinators

PhysioLab Intelligence is an interactive, browser-based computational physiology platform designed to bridge the gap between abstract textbook theory and signal processing practice. It provides students with real-time parameter manipulation, transparent multi-stage pipeline visualization, and randomized assessment modules without requiring heavy local software installations or risking patient privacy.

### 2. Core Learning Outcomes

By completing this practical session, students will be able to:

* **Electrocardiography (ECG):** Trace the origin of R-peak detection, understand the impact of signal noise, baseline wander, and mains power interference (50 Hz), and compute autonomic metrics (SDNN, RMSSD).
* **Electroencephalography (EEG):** Demonstrate the Berger effect (alpha wave attenuation during eye-opening) and interpret power spectral density (PSD) across frequency bands (Delta, Theta, Alpha, Beta).
* **Echocardiography (Echo):** Apply caliper geometry to left ventricular internal dimensions at end-diastole (LVIDd) and end-systole (LVIDs) to derive fractional shortening and estimate systolic function.

---

## PART 2: STUDENT PRACTICAL WORKSHEET

### Module A: ECG Signal Processing & Autonomic Variability

1. **Launch the Sandbox:** Navigate to the **Teaching lab** tab and select **ECG** > **Simulator (show the steps)**.
2. **Explore Pipeline Stages:** Use the stage selector to observe how raw data moves through:
* *Stage 1 (Raw Signal):* Observe environmental noise and drift.
* *Stage 2 (Band-Pass Filter):* Notice how the 5–15 Hz filter isolates QRS complex frequencies while stripping out baseline drift.
* *Stage 3 & 4 (Energy & R-Peaks):* Observe thresholding and automated peak detection.


3. **Parameter Sensitivity Exercise:**
* Adjust the **50 Hz Mains Interference** and **Baseline Wander** sliders. Record the exact threshold where the system flags signal quality degradation.
* Switch to **Guided exercises** to test your manual R-R interval counting against algorithmic output.



---

### Module B: EEG Alpha Rhythm & Spectral Analysis

1. **Select Modality:** In the **Teaching lab**, switch the modality toggle to **EEG**.
2. **Demonstrate the Berger Effect:**
* Set the state to **Eyes closed**. Observe the prominent 10 Hz synchronization on the **O1 (occipital)** channel.
* Switch the state to **Eyes open** and note the immediate alpha attenuation (alpha blocking).


3. **Spectral Decomposition:** Inspect the **Welch Power Spectral Density (PSD)** stage to analyze how signal power shifts across frequency bands when artifact noise is introduced.

---

### Module C: Echocardiography Ventricular Mechanics

1. **Select Modality:** In the **Teaching lab**, switch the modality toggle to **Echocardiography**.
2. **Caliper Geometry:** Adjust the **LVIDd** (End-Diastolic) and **LVIDs** (End-Systolic) sliders. Watch how the visual caliper overlay updates on the simulated 2D ventricular frame.
3. **Hemodynamic Derivation:** Examine how changes in internal dimensions alter the calculated **Fractional Shortening (FS)** and clinical status indicators.

---

## PART 3: ASSESSMENT & RESULT EXPORT

1. **Specialty Knowledge Quiz:**
* In the Teaching lab, select **Quiz** for your chosen specialty (ECG, EEG, or Echocardiography).
* Complete the 5 randomized questions drawn from the departmental question bank.
* Review instant explanations provided for every answer (correct or incorrect).


2. **Submission Protocol:**
* Upon completing the quiz, click **Download my results (CSV)**.
* Save your result file with your full name and student matriculation number (e.g., `quiz_ECG_StudentName.csv`).
* Submit the CSV file to your laboratory instructor for automated grading and attendance verification.



---

*Disclaimer: PhysioLab Intelligence is an educational and research-support platform. It is not cleared for primary clinical diagnosis.*