import json
from pathlib import Path
from physiolab.engines import ecg, eeg, echo
from physiolab.explain import explain, AUDIENCES

D = Path(__file__).parent.parent / "demo_data"


def test_ecg_hr_close_to_truth():
    sig, fs, _ = ecg.load_csv(D / "demo_ecg.csv")
    res, peaks = ecg.analyse(sig, fs)
    assert abs(res.measurements["heart_rate_bpm"] - 72) < 4
    assert res.quality["label"] in ("good", "fair")
    json.dumps(res.to_dict())


def test_ecg_flatline_is_poor():
    import numpy as np
    res, _ = ecg.analyse(np.zeros(5000), 250)
    assert res.quality["label"] == "poor"


def test_eeg_flags_bad_channels_and_alpha():
    data, fs, names = eeg.load_csv(D / "demo_eeg.csv")
    res = eeg.analyse(data, fs, names)
    assert "Fp2" in res.quality["flagged_channels"]
    assert "T7" in res.quality["flagged_channels"]
    assert 9 <= res.measurements["alpha_peak_hz"] <= 11


def test_echo_dicom_calibration_and_calipers():
    frames, meta, mm = echo.load_dicom((D / "demo_echo.dcm").read_bytes())
    assert frames.shape[0] == 10 and abs(mm - 0.3) < 1e-6
    assert "PatientName" not in meta
    res = echo.analyse(frames, meta, mm, [{"label": "LVIDd", "p1": (0, 0), "p2": (100, 0)},
                                          {"label": "LVIDs", "p1": (0, 0), "p2": (70, 0)}])
    assert res.measurements["caliper_mm"]["LVIDd"] == 30.0
    assert res.measurements["fractional_shortening_pct"] == 30.0


def test_ef_formula():
    assert echo.ejection_fraction(120, 50) == 58.3


def test_explanations_for_all_audiences():
    sig, fs, _ = ecg.load_csv(D / "demo_ecg.csv")
    r = ecg.analyse(sig, fs)[0].to_dict()
    for a in AUDIENCES:
        assert str(r["measurements"]["heart_rate_bpm"]) in explain(r, a)


def test_quality_detects_mains_and_wander():
    from physiolab.teaching import simulate_ecg
    _, clean, _ = simulate_ecg()
    _, bad, _ = simulate_ecg(mains=0.3, wander=0.5)
    qc = ecg.analyse(clean, 250)[0]
    qb = ecg.analyse(bad, 250)[0]
    assert qc.quality["score"] > qb.quality["score"] + 0.3
    assert any("Mains" in f for f in qb.flags) and any("wander" in f for f in qb.flags)


def test_simulator_heart_rate_tracks_setting():
    from physiolab.teaching import simulate_ecg, check_manual_hr
    for hr in (55, 90, 120):
        _, s, _ = simulate_ecg(hr=hr)
        assert abs(ecg.analyse(s, 250)[0].measurements["heart_rate_bpm"] - hr) < 4
    assert check_manual_hr(72, 74)["within_tolerance"]

def test_quiz_bank_and_scoring():
    from physiolab import quiz
    for spec in quiz.SPECIALTIES:
        assert len(quiz.BANK[spec]) >= quiz.QUIZ_LENGTH
        qs = quiz.draw(spec, seed=7)
        assert len(qs) == 5 and len({q["q"] for q in qs}) == 5
        for q in qs:
            assert 0 <= q["answer"] < len(q["options"]) and q["explanation"]
        n, rows = quiz.score(qs, [q["answer"] for q in qs])
        assert n == 5 and all(r["correct"] for r in rows)
        n, rows = quiz.score(qs, [None] * 5)
        assert n == 0 and all(r["explanation"] for r in rows)


def test_quiz_arithmetic_questions_are_correct():
    from physiolab import quiz
    assert 60000 / 800 == 75 and 8 * 6 == 48
    assert round((50 - 30) / 50 * 100) == 40 and round((120 - 50) / 120 * 100) == 58
    ecg_q = {q: (o, a) for q, o, a, _ in quiz.BANK["ECG"]}
    o, a = ecg_q["The mean R-R interval is 800 ms. What is the heart rate?"]
    assert o[a] == "75 bpm"
def test_validation_matching_logic():
    import numpy as np
    from validation.validate_ecg import match_beats
    ref = np.array([100, 300, 500, 700])
    assert match_beats(np.array([102, 298, 505, 698]), ref, 250) == (4, 0, 0)
    assert match_beats(np.array([102, 400, 698]), ref, 250) == (2, 1, 2)


def test_intervals_hidden_by_default():
    from physiolab.teaching import simulate_ecg
    _, s, _ = simulate_ecg()
    m = ecg.analyse(s, 250)[0].measurements
    assert "qrs_ms" not in m and "qt_ms" not in m

def test_eeg_simulator_alpha_blocking_and_artefacts():
    from physiolab import teaching as T
    def alpha(**kw):
        _, d, n, _ = T.simulate_eeg(**kw)
        return eeg.analyse(d, 250, n).measurements["global_relative_power"]["alpha"]
    assert alpha(eyes="closed") > alpha(eyes="open") + 0.2
    _, d, n, _ = T.simulate_eeg(flat_channel=True)
    assert eeg.analyse(d, 250, n).quality["flagged_channels"] == ["Fp2"]
    _, d, n, _ = T.simulate_eeg(mains=20)
    assert len(eeg.analyse(d, 250, n).quality["flagged_channels"]) >= 5


def test_eeg_dominant_band_exercise_is_decodable():
    from physiolab import teaching as T
    for band in ("delta", "theta", "alpha", "beta"):
        amps = {b: 5.0 for b in ("delta", "theta", "alpha", "beta")}; amps[band] = 30.0
        _, d, n, _ = T.simulate_eeg(amps=amps, alpha_posterior_only=False)
        g = eeg.analyse(d, 250, n).measurements["global_relative_power"]
        assert max(g, key=g.get) == band


def test_echo_simulator_measurement_matches_truth():
    from physiolab import teaching as T
    for d, s, mm in [(48, 32, 0.4), (56, 24, 0.6), (40, 26, 0.3)]:
        frames, truth = T.simulate_echo(d, s, mm)
        for f, true in zip(frames, (d, s)):
            x0, x1 = T.cavity_extent_px(f)
            assert abs((x1 - x0 + 1) * mm - true) <= 1.5 * mm + 0.5
    assert T.echo_fs(50, 30) == 40.0
    assert T.check_within(58.0, 58.3, 1.0)["within_tolerance"]    