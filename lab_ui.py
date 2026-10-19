"""Teaching lab interface (Streamlit). Specialty selector: ECG, EEG, Echocardiography.
Each specialty has a Simulator (show the steps), Guided exercises and a Quiz."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from physiolab import teaching as T
from physiolab import quiz as quizmod
from physiolab.engines import ecg, eeg, echo
from physiolab.explain import explain

FS = 250
BAND_COLOURS = {"delta": "#4c78a8", "theta": "#59a14f", "alpha": "#f28e2b", "beta": "#e15759", "gamma": "#b07aa1"}


def window_plot(t, y, a, b, peaks=None, thr=None, title=""):
    fig = go.Figure(go.Scatter(x=t[a:b], y=y[a:b], line=dict(width=1)))
    if thr is not None:
        fig.add_hline(y=thr, line_dash="dash", line_color="orange", annotation_text="threshold")
    if peaks is not None:
        pk = peaks[(peaks >= a) & (peaks < b)]
        fig.add_trace(go.Scatter(x=t[pk], y=y[pk], mode="markers", marker=dict(color="red", size=8), name="R-peaks"))
    fig.update_layout(height=300, title=title, xaxis_title="Time (s)", showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def spectrum_plot(f, p, title="Power spectrum (mean of usable channels)"):
    fig = go.Figure(go.Scatter(x=f, y=p, line=dict(width=1.5, color="#333")))
    for band, (lo, hi) in eeg.BANDS.items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor=BAND_COLOURS[band], opacity=0.15, line_width=0,
                      annotation_text=band, annotation_position="top left")
    fig.update_layout(height=320, title=title, xaxis_title="Frequency (Hz)", yaxis_title="Power", yaxis_type="log",
                      showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render_lab(audience):
    st.caption("Synthetic data only. Choose a specialty, then change a parameter and watch every stage respond.")
    c1, c2 = st.columns([1, 1.3])
    spec = c1.radio("Specialty", T.SPECIALTIES, horizontal=True, key="lab_spec")
    mode = c2.radio("Mode", ["Simulator (show the steps)", "Guided exercises", "Quiz"], horizontal=True, key="lab_mode")
    if mode == "Quiz":
        _quiz(spec)
    elif mode == "Guided exercises":
        _exercises(spec, audience)
    else:
        {"ECG": _sim_ecg, "EEG": _sim_eeg, "Echocardiography": _sim_echo}[spec](audience)


# ============================================================ SIMULATORS
def _sim_ecg(audience):
    c = st.columns(5)
    hr = c[0].slider("Heart rate (bpm)", 40, 160, 72, key="ecg_hr")
    var = c[1].slider("Variability (bpm SD)", 0.0, 10.0, 1.0, 0.5, key="ecg_var")
    noise = c[2].slider("Random noise", 0.0, 0.3, 0.03, 0.01, key="ecg_noise")
    wander = c[3].slider("Baseline wander (mV)", 0.0, 0.6, 0.0, 0.05, key="ecg_wander")
    mains = c[4].slider("50 Hz interference (mV)", 0.0, 0.4, 0.0, 0.05, key="ecg_mains")
    t, sig, _ = T.simulate_ecg(hr, 30, FS, noise, var, wander, mains)
    res, peaks = ecg.analyse(sig, FS, "simulated")
    st_ = ecg.detect_stages(sig, FS)
    stage = st.radio("Pipeline stage", ["1. Raw signal", "2. Band-pass filtered (5-15 Hz)",
                     "3. Energy and threshold", "4. Detected R-peaks"], horizontal=True, key="ecg_stage")
    a, b = 0, 10 * FS
    if stage.startswith("1"):
        st.plotly_chart(window_plot(t, st_["raw"], a, b, title="Raw signal"), use_container_width=True)
        st.caption("The recording as acquired, including any baseline wander, noise and mains interference.")
    elif stage.startswith("2"):
        st.plotly_chart(window_plot(t, st_["filtered"], a, b, title="Filtered: QRS emphasised, drift and mains reduced"), use_container_width=True)
        st.caption("A 5 to 15 Hz band-pass keeps the frequencies where QRS energy is concentrated.")
    elif stage.startswith("3"):
        st.plotly_chart(window_plot(t, st_["energy"], a, b, thr=st_["threshold"], title="Squared-derivative energy"), use_container_width=True)
        st.caption("The derivative is squared and smoothed over 80 ms. Peaks above the threshold become beat candidates.")
    else:
        st.plotly_chart(window_plot(t, st_["raw"], a, b, peaks=peaks, title="R-peaks on the original signal"), use_container_width=True)
        st.caption("Each candidate is refined to the true R-peak on the raw trace. R-R intervals then give heart rate and variability.")
    m = res.measurements
    cmp = pd.DataFrame({"Simulator setting": [hr, var, "-", "-"],
                        "Measured": [m.get("heart_rate_bpm", "-"), m.get("sdnn_ms", "-"), m.get("rmssd_ms", "-"), res.quality["label"]]},
                       index=["Heart rate (bpm)", "SDNN (ms) vs variability setting", "RMSSD (ms)", "Signal quality"])
    st.table(cmp.astype(str))
    for f in res.flags:
        st.warning(f)
    st.info(explain(res.to_dict(), audience))


def _sim_eeg(audience):
    c = st.columns(4)
    eyes = c[0].radio("Eyes", ["closed", "open"], horizontal=True, key="eeg_eyes")
    alpha = c[1].slider("Alpha amplitude (uV)", 0, 40, 25, key="eeg_alpha")
    theta = c[2].slider("Theta amplitude (uV)", 0, 30, 6, key="eeg_theta")
    beta = c[3].slider("Beta amplitude (uV)", 0, 30, 4, key="eeg_beta")
    c = st.columns(4)
    delta = c[0].slider("Delta amplitude (uV)", 0, 30, 3, key="eeg_delta")
    noise = c[1].slider("Random noise (uV)", 1, 20, 5, key="eeg_noise")
    mains = c[2].slider("50 Hz interference (uV)", 0, 30, 0, key="eeg_mains")
    flat = c[3].checkbox("Disconnect electrode Fp2", key="eeg_flat")
    t, data, names, _ = T.simulate_eeg(amps=dict(alpha=alpha, theta=theta, beta=beta, delta=delta), eyes=eyes,
                                       noise=noise, mains=mains, flat_channel=flat)
    res = eeg.analyse(data, FS, names, "simulated")
    pc = res.measurements["per_channel"]
    stage = st.radio("Analysis stage", ["1. Raw traces", "2. Power spectrum", "3. Band power and channel QC"],
                     horizontal=True, key="eeg_stage")
    if stage.startswith("1"):
        a, b = 0, 10 * FS
        fig = go.Figure()
        for i, ch in enumerate(names):
            seg = data[i, a:b]
            seg = (seg - seg.mean()) / (seg.std() + 1e-9) if pc[ch]["status"] != "flat" else seg * 0
            fig.add_trace(go.Scatter(x=t[a:b], y=seg * 0.4 - i, line=dict(width=1), name=ch))
        fig.update_layout(height=400, showlegend=False, xaxis_title="Time (s)",
                          yaxis=dict(tickvals=[-i for i in range(len(names))], ticktext=names), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Ten seconds of eight channels (each scaled for display). Look for a flat line or fast, regular ripples from mains interference.")
    elif stage.startswith("2"):
        rows = [i for i, ch in enumerate(names) if pc[ch]["status"] == "ok"]
        if rows:
            f, p = T.eeg_spectrum(data, FS, rows)
            st.plotly_chart(spectrum_plot(f, p), use_container_width=True)
            st.caption("Welch power spectrum. Shaded regions are the standard bands. A narrow spike at 50 Hz is mains interference, not brain activity.")
        else:
            st.warning("No usable channels remain, so there is no spectrum to show.")
    else:
        g = res.measurements["global_relative_power"]
        if g:
            st.plotly_chart(px.bar(x=list(g), y=list(g.values()), labels={"x": "Band", "y": "Relative power"},
                                   title="Global relative band power (usable channels)").update_layout(height=280), use_container_width=True)
        else:
            st.warning("Every channel was flagged, so no band-power summary can be given.")
        st.dataframe(pd.DataFrame(pc).T[["status", "std_uV", "rel_delta", "rel_theta", "rel_alpha", "rel_beta"]])
    for f_ in res.flags:
        st.warning(f_)
    st.info(explain(res.to_dict(), audience))


def _echo_figure(frame, ext, row, label):
    fig = px.imshow(frame, color_continuous_scale="gray", aspect="equal", title=label)
    if ext:
        fig.add_shape(type="line", x0=ext[0], x1=ext[1] + 1, y0=row, y1=row, line=dict(color="red", width=3))
    fig.update_layout(coloraxis_showscale=False, height=360, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _sim_echo(audience):
    c = st.columns(3)
    lvidd = c[0].slider("LVIDd (mm)", 35, 70, 48, key="echo_lvidd")
    lvids = c[1].slider("LVIDs (mm)", 20, 55, 32, key="echo_lvids")
    mm = c[2].slider("Calibration (mm per pixel)", 0.2, 0.6, 0.4, 0.05, key="echo_mm")
    if lvids >= lvidd:
        st.warning("LVIDs must be smaller than LVIDd.")
        return
    frames, truth = T.simulate_echo(lvidd, lvids, mm)
    row = truth["row"]
    exts = [T.cavity_extent_px(f) for f in frames]
    col1, col2 = st.columns(2)
    col1.plotly_chart(_echo_figure(frames[0], exts[0], row, "End-diastole"), use_container_width=True)
    col2.plotly_chart(_echo_figure(frames[1], exts[1], row, "End-systole"), use_container_width=True)
    if not all(exts):
        st.warning("The cavity could not be measured at these settings.")
        return
    px_d, px_s = exts[0][1] - exts[0][0] + 1, exts[1][1] - exts[1][0] + 1
    calipers = [{"label": "LVIDd", "p1": (exts[0][0], row), "p2": (exts[0][1] + 1, row)},
                {"label": "LVIDs", "p1": (exts[1][0], row), "p2": (exts[1][1] + 1, row)}]
    res = echo.analyse(np.stack(frames), {"Modality": "simulated"}, mm, calipers, "simulated")
    m = res.measurements
    steps = pd.DataFrame({
        "End-diastole (LVIDd)": [px_d, mm, m["caliper_mm"]["LVIDd"]],
        "End-systole (LVIDs)": [px_s, mm, m["caliper_mm"]["LVIDs"]]},
        index=["1. Cavity width (pixels)", "2. Calibration (mm per pixel)", "3. Width = pixels x calibration (mm)"])
    st.table(steps.astype(str))
    st.metric("4. Fractional shortening (%)", m.get("fractional_shortening_pct", "-"),
              help="(LVIDd - LVIDs) / LVIDd x 100")
    st.caption("Without the calibration in step 2, only pixel counts exist, and they have no physical meaning.")
    st.info(explain(res.to_dict(), audience))


# ============================================================ EXERCISES
def _log(spec, idx):
    cols = {"ECG": ["Setting", "Heart rate", "SDNN", "RMSSD", "Quality", "Observation"],
            "EEG": ["Setting", "Global alpha power", "Dominant band", "Flagged channels", "Observation"],
            "Echocardiography": ["Setting", "mm per pixel", "Width (px)", "LVIDd (mm)", "Observation"]}[spec]
    st.info("Use the Simulator mode for this exercise and record your values below.")
    df = pd.DataFrame({c: ["", "", ""] for c in cols})
    log = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"log_{spec}_{idx}")
    st.download_button("Download my results (CSV)", log.to_csv(index=False), f"lab_results_{spec}.csv", "text/csv",
                       key=f"dl_{spec}_{idx}")


def _exercises(spec, audience):
    exs = T.EXERCISES[spec]
    names = list(exs)
    ex = st.selectbox("Exercise", names, key=f"ex_{spec}")
    idx = names.index(ex)
    st.markdown(f"**Aim:** {exs[ex]['aim']}")
    for i, step in enumerate(exs[ex]["steps"], 1):
        st.markdown(f"{i}. {step}")
    if spec == "ECG" and idx == 0:
        _ecg_ex1()
    elif spec == "EEG" and idx == 0:
        _eeg_ex1()
    elif spec == "Echocardiography" and idx == 0:
        _echo_ex1()
    elif spec == "Echocardiography" and idx == 1:
        _echo_ex2()
    else:
        _log(spec, idx)


def _ecg_ex1():
    if "ex1" not in st.session_state or st.button("New trace", key="ecg_new"):
        rng = np.random.default_rng()
        st.session_state.ex1 = {"hr": int(rng.integers(50, 111)), "seed": int(rng.integers(1, 10_000)), "shown": False}
    e = st.session_state.ex1
    t, sig, _ = T.simulate_ecg(e["hr"], 30, FS, 0.03, 1.0, 0, 0, e["seed"])
    st.plotly_chart(window_plot(t, sig, 0, 10 * FS, title="10-second window: count the QRS complexes"), use_container_width=True)
    guess = st.number_input("Your estimate (bpm)", 20, 250, 70, key="ecg_guess")
    if st.button("Check my answer", key="ecg_check"):
        e["shown"] = True
    if e["shown"]:
        r, _ = ecg.analyse(sig, FS, "exercise")
        comp = r.measurements["heart_rate_bpm"]
        chk = T.check_manual_hr(guess, comp)
        st.metric("Algorithm result (bpm)", comp, delta=chk["error_bpm"], delta_color="off")
        (st.success if chk["within_tolerance"] else st.warning)(f"{chk['message']} Your error: {chk['error_bpm']} bpm.")


def _eeg_ex1():
    bands = ["delta", "theta", "alpha", "beta"]
    if "eeg_ex1" not in st.session_state or st.button("New trace", key="eeg_new"):
        rng = np.random.default_rng()
        st.session_state.eeg_ex1 = {"band": str(rng.choice(bands)), "seed": int(rng.integers(1, 10_000)), "shown": False}
    e = st.session_state.eeg_ex1
    amps = {b: 5.0 for b in bands}
    amps[e["band"]] = 30.0
    amps["gamma"] = 0.0
    t, data, names, _ = T.simulate_eeg(amps=amps, alpha_posterior_only=False, seed=e["seed"])
    n = 5 * FS
    fig = go.Figure(go.Scatter(x=t[:n], y=data[names.index("O1"), :n], line=dict(width=1.2)))
    fig.update_layout(height=280, title="Channel O1, 5 seconds: count the peaks in one second", xaxis_title="Time (s)",
                      yaxis_title="uV", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    options = ["delta (0.5-4 Hz)", "theta (4-8 Hz)", "alpha (8-13 Hz)", "beta (13-30 Hz)"]
    guess = st.radio("Which band dominates this trace?", options, index=None, key="eeg_guess")
    if st.button("Check my answer", key="eeg_check"):
        e["shown"] = True
    if e["shown"]:
        if guess is None:
            st.warning("Choose a band first.")
            return
        res = eeg.analyse(data, FS, names, "exercise")
        g = res.measurements["global_relative_power"]
        dom = max(g, key=g.get)
        (st.success if guess.startswith(dom) else st.error)(
            f"{'Correct' if guess.startswith(dom) else 'Not quite'}: the dominant band is {dom} ({g[dom] * 100:.0f}% of power).")
        f, p = T.eeg_spectrum(data, FS)
        st.plotly_chart(spectrum_plot(f, p, "Power spectrum"), use_container_width=True)
        st.info("Rule of thumb: about 2 peaks per second is delta, 6 is theta, 10 is alpha, 20 is beta. "
                "The spectrum shows the same thing quantitatively.")


def _echo_ex1():
    if "echo_ex1" not in st.session_state or st.button("New case", key="echo_new"):
        rng = np.random.default_rng()
        d = int(rng.integers(42, 58))
        s = int(rng.integers(24, min(40, d - 8)))
        st.session_state.echo_ex1 = {"d": d, "s": s, "seed": int(rng.integers(1, 10_000)), "shown": False}
        for k in ("echo1_dl", "echo1_dr", "echo1_sl", "echo1_sr"):
            st.session_state.pop(k, None)
    e = st.session_state.echo_ex1
    mm = 0.4
    frames, truth = T.simulate_echo(e["d"], e["s"], mm, seed=e["seed"])
    row = truth["row"]
    st.caption(f"Calibration: {mm} mm per pixel. Drag the sliders so the red line spans the dark cavity.")
    meas = {}
    colA, colB = st.columns(2)
    for col, frame, label, kl, kr in ((colA, frames[0], "End-diastole", "echo1_dl", "echo1_dr"),
                                     (colB, frames[1], "End-systole", "echo1_sl", "echo1_sr")):
        with col:
            xl = st.slider(f"{label}: left caliper (px)", 0, 299, 100, key=kl)
            xr = st.slider(f"{label}: right caliper (px)", 0, 299, 200, key=kr)
            fig = px.imshow(frame, color_continuous_scale="gray", aspect="equal", title=label)
            fig.add_shape(type="line", x0=xl, x1=xr, y0=row, y1=row, line=dict(color="red", width=3))
            fig.update_layout(coloraxis_showscale=False, height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            meas[label] = round(abs(xr - xl) * mm, 1)
    st.write(f"Your measurements: LVIDd {meas['End-diastole']} mm, LVIDs {meas['End-systole']} mm.")
    if st.button("Check my measurements", key="echo_check"):
        e["shown"] = True
    if e["shown"]:
        d_err = T.check_within(meas["End-diastole"], e["d"], 2.0)
        s_err = T.check_within(meas["End-systole"], e["s"], 2.0)
        ok = d_err["within_tolerance"] and s_err["within_tolerance"]
        fs_you = T.echo_fs(meas["End-diastole"], meas["End-systole"]) if meas["End-diastole"] > 0 else float("nan")
        (st.success if ok else st.warning)(
            "Both measurements are within 2 mm of the reference." if ok else
            "At least one measurement is more than 2 mm from the reference. Adjust the caliper ends to the cavity edges.")
        st.table(pd.DataFrame({"You": [meas["End-diastole"], meas["End-systole"], fs_you],
                               "Reference": [e["d"], e["s"], truth["fractional_shortening_pct"]]},
                              index=["LVIDd (mm)", "LVIDs (mm)", "Fractional shortening (%)"]).astype(str))


def _echo_ex2():
    if "echo_ex2" not in st.session_state or st.button("New volumes", key="ef_new"):
        rng = np.random.default_rng()
        st.session_state.echo_ex2 = {"edv": int(rng.integers(90, 151)), "esv": int(rng.integers(35, 71)), "shown": False}
    e = st.session_state.echo_ex2
    st.markdown(f"**End-diastolic volume:** {e['edv']} mL  \n**End-systolic volume:** {e['esv']} mL")
    guess = st.number_input("Your ejection fraction (%)", 0.0, 100.0, 50.0, 0.1, key="ef_guess")
    if st.button("Check my answer", key="ef_check"):
        e["shown"] = True
    if e["shown"]:
        true = echo.ejection_fraction(e["edv"], e["esv"])
        chk = T.check_within(guess, true, 1.0)
        (st.success if chk["within_tolerance"] else st.warning)(
            f"{'Correct' if chk['within_tolerance'] else 'Not quite'}: EF = {true}% (your error {chk['error']} points).")
        st.info(f"Stroke volume = EDV - ESV = {e['edv'] - e['esv']} mL. "
                f"EF = ({e['edv']} - {e['esv']}) / {e['edv']} x 100 = {true}%. "
                "Interpreting whether this value is normal is a clinical judgement.")


# ============================================================ QUIZ
def _quiz(spec):
    st.caption("Five questions drawn from a larger bank. Explanations appear after you submit, for every question.")
    qz = st.session_state.get("quiz")
    if qz is None or qz["spec"] != spec or st.button("New quiz", key="quiz_new"):
        seed = int(np.random.default_rng().integers(1, 1_000_000))
        st.session_state.quiz = qz = {"spec": spec, "seed": seed, "qs": quizmod.draw(spec, seed=seed), "result": None}
    with st.form(f"quizform_{qz['seed']}"):
        picks = [st.radio(f"**{i + 1}. {q['q']}**", q["options"], index=None, key=f"q{qz['seed']}_{i}")
                 for i, q in enumerate(qz["qs"])]
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        if any(p is None for p in picks):
            st.warning("Please answer all five questions before submitting.")
        else:
            chosen = [q["options"].index(p) for q, p in zip(qz["qs"], picks)]
            qz["result"] = quizmod.score(qz["qs"], chosen)
    if qz["result"]:
        n_ok, rows = qz["result"]
        st.subheader(f"Score: {n_ok} / {len(rows)}")
        for i, r in enumerate(rows, 1):
            (st.success if r["correct"] else st.error)(
                f"Q{i}: {'Correct' if r['correct'] else 'Not quite'}. Your answer: {r['your_answer']}")
            st.markdown(f"**Correct answer:** {r['correct_answer']}")
            st.info(r["explanation"])
        st.download_button("Download my results (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           f"quiz_{spec}.csv", "text/csv", key="quiz_dl")