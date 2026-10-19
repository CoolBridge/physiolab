"""Specialty quizzes (ECG, EEG, Echocardiography). Five questions per attempt, drawn at random
from a larger bank. Explanations are shown after submission for every question, right or wrong."""
import random

SPECIALTIES = ["ECG", "EEG", "Echocardiography"]
QUIZ_LENGTH = 5

# Each item: (question, [options], index_of_correct_option, explanation)
BANK = {
    "ECG": [
        ("The mean R-R interval is 800 ms. What is the heart rate?",
         ["60 bpm", "75 bpm", "80 bpm", "100 bpm"], 1,
         "Heart rate = 60,000 ms per minute divided by the R-R interval in ms. 60,000 / 800 = 75 bpm."),
        ("A 10-second strip contains 8 QRS complexes. What is the approximate heart rate?",
         ["48 bpm", "60 bpm", "80 bpm", "100 bpm"], 0,
         "Multiply the number of QRS complexes in a 10-second window by 6: 8 x 6 = 48 bpm. "
         "This is the hand-count method used in Guided Exercise 1."),
        ("What does SDNN describe?",
         ["The average heart rate", "The overall variability of normal R-R intervals",
          "The width of the QRS complex", "The length of the PR interval"], 1,
         "SDNN is the standard deviation of normal-to-normal (R-R) intervals. It reflects overall heart rate variability, "
         "combining sympathetic and parasympathetic influences over the recording."),
        ("Which measure mainly reflects short-term, beat-to-beat (vagally mediated) variability?",
         ["RMSSD", "Mean heart rate", "QT interval", "Signal-to-noise ratio"], 0,
         "RMSSD is the root mean square of successive R-R differences. It is sensitive to rapid parasympathetic (vagal) changes "
         "and is widely used for short recordings."),
        ("Why is a 5 to 15 Hz band-pass filter used before detecting R-peaks?",
         ["QRS energy is concentrated there, and drift and high-frequency noise are reduced",
          "It makes the P wave larger", "It converts the signal to heart rate", "It removes the need for a threshold"], 0,
         "Most QRS energy lies in roughly this range. Band-passing attenuates baseline wander (very low frequencies) and "
         "muscle or electrical noise (higher frequencies), so QRS complexes stand out."),
        ("A regular sinusoidal 50 Hz pattern is superimposed on an ECG recorded in Nigeria. What is the most likely cause?",
         ["Mains electrical interference", "Patient breathing", "A tall T wave", "Low sampling rate"], 0,
         "Nigeria's mains supply is 50 Hz, so interference at exactly 50 Hz points to the power line. Good grounding, shielded leads "
         "and a notch filter reduce it."),
        ("What is the usual normal range of the PR interval in adults?",
         ["20 to 80 ms", "120 to 200 ms", "300 to 400 ms", "500 to 700 ms"], 1,
         "The PR interval, from the start of atrial depolarisation to the start of ventricular depolarisation, is normally "
         "about 120 to 200 ms. In this platform PR estimates are approximate until validated."),
    ],
    "EEG": [
        ("Which frequency range defines the alpha band?",
         ["0.5 to 4 Hz", "4 to 8 Hz", "8 to 13 Hz", "13 to 30 Hz"], 2,
         "Alpha is 8 to 13 Hz. It is typically strongest over posterior regions in a relaxed, awake, eyes-closed state."),
        ("Which band dominates the slow-wave activity of deep (N3) sleep?",
         ["Gamma", "Beta", "Alpha", "Delta"], 3,
         "Delta (about 0.5 to 4 Hz) high-amplitude slow waves characterise N3, also called slow-wave sleep."),
        ("A strong, narrow spectral peak at exactly 50 Hz in an EEG recorded in Nigeria most likely reflects what?",
         ["Gamma-band cognition", "Mains interference", "Eye blinks", "Alpha rhythm"], 1,
         "A sharp peak at the mains frequency is environmental noise, not brain activity. Check grounding, electrode impedance "
         "and cable shielding before interpreting any gamma-range findings."),
        ("What happens to the posterior alpha rhythm when a relaxed person opens their eyes?",
         ["It increases", "It is attenuated (alpha blocking)", "It becomes delta", "It is unchanged"], 1,
         "Eye opening and visual attention suppress posterior alpha, known as alpha blocking or the Berger effect. "
         "It is a classic demonstration in physiology practicals."),
        ("A channel shows an almost perfectly flat line. What is the most sensible first conclusion?",
         ["The person has no brain activity", "A disconnected or faulty electrode or connection",
          "Very strong alpha", "Normal sleep"], 1,
         "A flat channel usually indicates a technical problem such as a lifted electrode, broken lead or reference fault. "
         "Brain death is not diagnosed from a single flat channel."),
        ("What does relative band power tell you?",
         ["The absolute amplitude in microvolts", "The fraction of total power that lies within a band",
          "The number of electrodes used", "The sampling rate"], 1,
         "Relative power divides band power by total power, so values sum to about 1. This makes comparisons less dependent "
         "on overall signal amplitude."),
        ("Why is Welch's method used to estimate the power spectrum?",
         ["It averages spectra from overlapping segments, reducing variance",
          "It removes all artefacts", "It increases the sampling rate", "It identifies seizures"], 0,
         "Welch's method splits the signal into overlapping windows and averages their spectra, giving a smoother, more stable "
         "estimate than one raw periodogram. It does not remove artefacts or detect disease."),
    ],
    "Echocardiography": [
        ("LVIDd is 50 mm and LVIDs is 30 mm. What is the fractional shortening?",
         ["20%", "30%", "40%", "60%"], 2,
         "FS = (LVIDd - LVIDs) / LVIDd x 100 = (50 - 30) / 50 x 100 = 40%. It is a simple linear index of left ventricular systolic function."),
        ("Why must an echo image be calibrated before measuring distances?",
         ["To improve the colours", "To convert pixel distances into millimetres",
          "To increase the frame rate", "To remove the need for a probe"], 1,
         "Calipers measure in pixels. The scale (millimetres per pixel) comes from the DICOM ultrasound region or pixel spacing, "
         "and without it a distance has no physical meaning."),
        ("Which standard view is commonly used for linear measurements of LVIDd and LVIDs?",
         ["Parasternal long axis", "Subcostal inferior vena cava view", "Suprasternal view", "Apical five-chamber view"], 0,
         "Linear LV dimensions are classically measured from the parasternal long-axis view, perpendicular to the LV long axis "
         "at the level of the mitral leaflet tips."),
        ("End-diastolic volume is 120 mL and end-systolic volume is 50 mL. What is the ejection fraction?",
         ["42%", "58%", "70%", "83%"], 1,
         "EF = (EDV - ESV) / EDV x 100 = (120 - 50) / 120 x 100 = 58.3%, about 58%."),
        ("At which point in the cardiac cycle is the LV internal dimension largest, giving LVIDd?",
         ["End-systole", "End-diastole", "Mid-systole", "Isovolumic relaxation only"], 1,
         "The ventricle is most filled at end-diastole, so LVIDd is measured then, while LVIDs is measured at the smallest dimension "
         "in end-systole."),
        ("Why does this platform not report a diagnosis from caliper measurements?",
         ["Measurements depend on image quality, caliper placement and clinical context, so interpretation needs a clinician",
          "Calipers are always inaccurate", "Echo cannot measure the heart", "Diagnosis is illegal"], 0,
         "A distance or index is only one input. Image quality, technique, reference ranges and the whole clinical picture "
         "are needed, which is why interpretation stays with the reporting clinician."),
        ("What physical principle forms an echocardiogram?",
         ["X-ray absorption", "Reflection of high-frequency sound waves from tissue interfaces",
          "Nuclear decay", "Magnetic resonance of protons"], 1,
         "Ultrasound is emitted by the probe and reflected at boundaries between tissues of different acoustic impedance. "
         "The returning echoes are timed and displayed as an image. No ionising radiation is used."),
    ],
}


def draw(specialty: str, n: int = QUIZ_LENGTH, seed=None):
    """Return n shuffled questions: [{'q', 'options', 'answer', 'explanation'}]."""
    rng = random.Random(seed)
    items = rng.sample(BANK[specialty], min(n, len(BANK[specialty])))
    out = []
    for q, opts, ans, expl in items:
        order = list(range(len(opts)))
        rng.shuffle(order)
        out.append({"q": q, "options": [opts[i] for i in order], "answer": order.index(ans), "explanation": expl})
    return out


def score(questions, chosen):
    """chosen: list of selected option indices (None if unanswered). Returns (correct_count, per_question rows)."""
    rows, correct = [], 0
    for q, c in zip(questions, chosen):
        ok = c is not None and c == q["answer"]
        correct += ok
        rows.append({"question": q["q"], "your_answer": q["options"][c] if c is not None else "(no answer)",
                     "correct_answer": q["options"][q["answer"]], "correct": ok, "explanation": q["explanation"]})
    return correct, rows