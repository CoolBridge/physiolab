import io, json, os, base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from physiolab import db
from physiolab.engines import ecg, eeg, echo
from physiolab.explain import explain, AUDIENCES
from physiolab.schema import DISCLAIMER
from physiolab.teaching import simulate_ecg, simulate_eeg_pipeline, simulate_echo_pipeline, check_manual_hr, EXERCISES
from physiolab import quiz as quizmod
from supabase import create_client, Client

st.set_page_config(page_title="PhysioLab Intelligence", page_icon="assets/logo1.jpeg", layout="wide")
DEMO = "demo_data"

# ---------- Supabase Client Initialization via secrets.toml ----------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# ---------- Helper to Base64 Encode Assets for CSS & HTML ----------
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            mime = "image/png" if image_path.endswith(".png") else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
    return ""

hero_bg_data = get_base64_image("assets/hero_display.jpeg")

# Find available logo for the hero and sidebar
logo_path_found = None
logo_mime_data = ""
for candidate in ["assets/logo1.jpeg", "assets/logo1.jpg", "assets/logo1.png", "assets/logq1.png"]:
    if os.path.exists(candidate):
        logo_path_found = candidate
        logo_mime_data = get_base64_image(candidate)
        break

# ---------- Cinematic Dark Theme & Edge-to-Edge Footer CSS ----------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0b071e !important;
        color: #f3f4f6;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    .block-container {{
        padding-top: 0rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Cinematic Hero Section with Embedded Base64 Background */
    .hero-fullscreen {{
        position: relative;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        background: linear-gradient(135deg, rgba(11, 7, 30, 0.82) 0%, rgba(23, 16, 59, 0.88) 100%),
                    url('{hero_bg_data}');
        background-size: cover;
        background-position: center;
        border-bottom: 2px solid #281d54;
        padding: 4rem 5rem;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
    }}

    .hero-logo {{
        width: 64px;
        height: 64px;
        border-radius: 12px;
        object-fit: cover;
        margin-bottom: 1.2rem;
        border: 2px solid rgba(14, 165, 233, 0.4);
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }}

    .hero-subtitle {{
        color: #0ea5e9;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}

    .hero-title {{
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #ffffff;
        margin-bottom: 1.5rem;
        max-width: 900px;
    }}

    .hero-description {{
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 750px;
        margin-bottom: 2rem;
        line-height: 1.6;
    }}

    /* Floating Modality Cards & High-Contrast White Content Frame */
    div[data-testid="stExpander"] {{
        background-color: #130b2e !important;
        border: 1px solid #3b2a73 !important;
        border-radius: 12px !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }}
    div[data-testid="stExpander"]:hover {{
        border-color: #0ea5e9 !important;
        box-shadow: 0 15px 35px rgba(14, 165, 233, 0.25);
    }}
    
    div[data-testid="stExpander"] summary, 
    div[data-testid="stExpander"] summary span, 
    div[data-testid="stExpander"] summary p,
    .streamlit-expanderHeader, 
    .streamlit-expanderHeader p, 
    .streamlit-expanderHeader span {{
        background-color: #130b2e !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        border-radius: 12px !important;
    }}

    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {{
        background-color: #ffffff !important;
        color: #0f172a !important;
        padding: 1.5rem !important;
        border-bottom-left-radius: 12px;
        border-bottom-right-radius: 12px;
    }}
    
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] p, 
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] b, 
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] li {{
        color: #1e293b !important;
        line-height: 1.6;
    }}

    [data-testid="stSidebar"] {{
        background-color: #0d0824;
        border-right: 1px solid #281d54;
    }}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {{
        color: #f8fafc !important;
    }}

    div.stMetric {{
        background-color: #150f38;
        border: 1px solid #281d54;
        border-radius: 8px;
        padding: 1rem;
    }}

    .footer-container {{
        position: relative;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
        margin-top: 5rem;
        margin-bottom: 0rem !important;
        padding: 1.5rem 2rem;
        border-top: 1px solid #281d54;
        background-color: #0d0824;
        color: #94a3b8;
        font-size: 0.8rem;
        text-align: center;
    }}
    .footer-brand {{
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Navigation & User Session
if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "Home"

if "user" not in st.session_state:
    st.session_state.user = None

if "intro_watched" not in st.session_state:
    st.session_state.intro_watched = False

# ---------- Cinematic Hero Header Render with Embedded Logo ----------
logo_img_tag = f'<img src="{logo_mime_data}" class="hero-logo">' if logo_mime_data else ''

st.markdown(f"""
<div class="hero-fullscreen">
    {logo_img_tag}
    <div class="hero-subtitle">MULTIMODAL PHYSIOLOGICAL DATA PLATFORM · V0.2</div>
    <div class="hero-title">ADVANCED COMPUTATIONAL PHYSIOLOGY & RESEARCH INTELLIGENCE</div>
    <div class="hero-description">
        Bridging the gap between abstract biomedical theory and practical computational analysis. Designed specifically for university physiology departments and cardiology research units.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- First-Time Visitor Biomimetic Intro Banner Card ----------
if not st.session_state.intro_watched:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #181042 0%, #291b66 100%); border: 2px solid #0ea5e9; border-radius: 16px; padding: 2rem; margin-bottom: 2.5rem; box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25);">
        <h3 style="color: #ffffff; margin-top: 0; font-size: 1.5rem;">🫀 Biomimetic Intelligence Preview</h3>
        <p style="color: #38bdf8; font-size: 0.95rem; margin-bottom: 1rem;">AI, Physiology, and Biomimetics in Cardiac Research: Inspired by Nature, Engineered by AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="position: relative; padding-bottom: 45%; height: 0; overflow: hidden; border-radius: 12px; margin-bottom: 1.5rem; border: 2px solid #0ea5e9;">
            <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                src="https://www.youtube.com/embed/js-n0USt25A?autoplay=1&mute=1&start=0&end=5&loop=1&playlist=js-n0USt25A&controls=0&modestbranding=1" 
                title="Biomimetic Artificial Heart" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_ib1, col_ib2 = st.columns(2)
    with col_ib1:
        if st.button("🚀 Enter PhysioLab Platform", use_container_width=True, type="primary"):
            st.session_state.intro_watched = True
            st.rerun()
    with col_ib2:
        if st.button("▶ Watch Full Video on YouTube", use_container_width=True):
            st.session_state.intro_watched = True
            st.markdown("[Open full video here](https://www.youtube.com/watch?v=js-n0USt25A)")
            st.rerun()
    st.divider()

# ---------- Navigation Bar Without Admin ----------
tabs_list = ["Home", "Analyse", "Teaching Lab", "About"]
selected_tab = st.radio("Navigation Menu", tabs_list, index=tabs_list.index(st.session_state.nav_tab) if st.session_state.nav_tab in tabs_list else 0, horizontal=True, key="main_nav_radio")

if selected_tab != st.session_state.nav_tab:
    st.session_state.nav_tab = selected_tab
    st.rerun()

st.divider()

# ---------- Sidebar Access & Session State ----------
with st.sidebar:
    if logo_path_found:
        st.image(logo_path_found, width=130)
    else:
        st.markdown("### 🫀 PhysioLab")
        
    st.header("Access & Session")
    if st.session_state.user:
        st.success(f"Signed in: {st.session_state.user}")
        if st.button("Sign out", key="btn_signout"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.user = None
            st.rerun()
    else:
        st.info("Not signed in")
    audience = st.selectbox("Explanation mode", AUDIENCES, index=3)
    st.divider()
    st.caption("Use only de-identified or synthetic data. In-memory processing only.")

# ---------- Helper Functions ----------
def finish(res, extra=None):
    d = res.to_dict()
    db.log_event(st.session_state.user, d["modality"], "success")
    st.subheader("Explanation")
    st.info(explain(d, audience))
    c1, c2 = st.columns(2)
    c1.metric("Quality", d["quality"].get("label", "-"))
    for f in d["flags"]:
        st.warning(f)
    with st.expander("Structured results (what the explanation is built from)", expanded=False):
        st.json(d)
    with st.expander("Limitations & provenance"):
        st.write(d["limitations"]); st.json(d["provenance"])
    flat = pd.json_normalize(d["measurements"], sep=".").T.reset_index()
    flat.columns = ["measurement", "value"]
    b1, b2 = st.columns(2)
    b1.download_button("Download JSON", json.dumps(d, indent=2), f"{d['modality']}_result.json", "application/json", key="dl_json")
    b2.download_button("Download CSV", flat.to_csv(index=False), f"{d['modality']}_result.csv", "text/csv", key="dl_csv")

def get_bytes(label, types, demo_file):
    up = st.file_uploader(label, type=types, key=f"upl_{demo_file}")
    if st.button(f"Load demo file ({demo_file})", key=f"demo_btn_{demo_file}"):
        st.session_state["demo_" + demo_file] = True
    if up:
        return up.getvalue(), up.name
    if st.session_state.get("demo_" + demo_file):
        return open(f"{DEMO}/{demo_file}", "rb").read(), demo_file
    return None, None

def window_plot(t, y, a, b, peaks=None, thr=None, title=""):
    fig = go.Figure(go.Scatter(x=t[a:b], y=y[a:b], line=dict(width=1, color="#0ea5e9")))
    if thr is not None:
        fig.add_hline(y=thr, line_dash="dash", line_color="orange", annotation_text="threshold")
    if peaks is not None:
        pk = peaks[(peaks >= a) & (peaks < b)]
        fig.add_trace(go.Scatter(x=t[pk], y=y[pk], mode="markers", marker=dict(color="#f43f5e", size=8), name="R-peaks"))
    fig.update_layout(height=300, title=title, xaxis_title="Time (s)", showlegend=False, 
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6"),
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---------- Routing Views based on st.session_state.nav_tab ----------
if st.session_state.nav_tab == "Home":
    if not st.session_state.user:
        st.subheader("🔒 Platform Authentication Required")
        st.caption("Sign in to your account or register below to unlock analytical modules, simulation labs, and assessment engines.")
        
        auth_mode = st.radio("Select Action", ["Sign In", "Create Account"], horizontal=True, key="home_auth_mode")
        
        if auth_mode == "Create Account":
            with st.form("supabase_signup_form"):
                c1, c2 = st.columns(2)
                reg_email = c1.text_input("Email Address*")
                reg_password = c2.text_input("Password (min 6 chars)*", type="password")
                
                c3, c4 = st.columns(2)
                reg_name = c3.text_input("Full Name*")
                reg_inst = c4.text_input("Institution*")
                
                if st.form_submit_button("Create Account"):
                    if reg_email and reg_password and reg_name:
                        try:
                            response = supabase.auth.sign_up({
                                "email": reg_email,
                                "password": reg_password,
                                "options": {
                                    "data": {
                                        "full_name": reg_name,
                                        "institution": reg_inst
                                    }
                                }
                            })
                            st.success("Account created successfully! Please check your email for confirmation if required.")
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
                    else:
                        st.warning("Please fill in all required fields.")
        else:
            with st.form("supabase_signin_form"):
                login_email = st.text_input("Email Address")
                login_password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Sign In"):
                    if login_email and login_password:
                        try:
                            response = supabase.auth.sign_in_with_password({
                                "email": login_email,
                                "password": login_password
                            })
                            st.session_state.user = login_email.strip().lower()
                            st.success("Successfully signed in!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sign-in failed. Please check your credentials or confirm your email: {e}")
                    else:
                        st.warning("Please enter both email and password.")
            
            with st.expander("Forgot your password?"):
                with st.form("forgot_password_form"):
                    reset_email = st.text_input("Enter your registered email")
                    if st.form_submit_button("Send Password Reset Email"):
                        if reset_email:
                            try:
                                supabase.auth.reset_password_for_email(reset_email)
                                st.success("Password reset instructions have been sent to your email!")
                            except Exception as e:
                                st.error(f"Could not send reset email: {e}")
                        else:
                            st.warning("Please enter your email address.")
    else:
        st.success(f"Welcome back, {st.session_state.user}! You have full access to all analytical modules and teaching labs.")
        
    st.markdown("---")
    st.subheader("Explore Modalities & Core Contributions")
    st.markdown('<p style="color: #ffffff; font-size: 1rem; margin-bottom: 1.5rem;">Click any floating modality card below to inspect its core study parameters and teaching guidelines in high contrast.</p>', unsafe_allow_html=True)
    
    col_e, col_ee, col_ec = st.columns(3)
    
    with col_e:
        with st.expander("⚡ Electrocardiography (ECG)"):
            st.markdown("### ⚡ Electrocardiography (ECG)")
            st.markdown("**Study & Practice:** Measures electrical potentials generated by heart muscle depolarization and repolarization.")
            st.markdown("**Cardiology Teaching:** Crucial for teaching automated R-peak detection, QRS complex morphology, PR/QT intervals, and autonomic heart rate variability (HRV via SDNN and RMSSD).")
            
    with col_ee:
        with st.expander("🧠 Electroencephalography (EEG)"):
            st.markdown("### 🧠 Electroencephalography (EEG)")
            st.markdown("**Study & Practice:** Records spontaneous electrical activity of the brain via scalp electrodes.")
            st.markdown("**Physiology Teaching:** Demonstrates cortical reactivity, such as the Berger effect (alpha wave attenuation during eye-opening), and Welch Power Spectral Density (PSD) across Delta, Theta, Alpha, and Beta frequency bands.")
            
    with col_ec:
        with st.expander("🫀 Echocardiography (Echo)"):
            st.markdown("### 🫀 Echocardiography (Echo)")
            st.markdown("**Study & Practice:** Ultrasound imaging to visualize myocardial structure, valve motion, and blood flow hemodynamics.")
            st.markdown("**Clinical Teaching:** Teaches ventricular caliper measurement, end-diastolic (LVIDd) vs. end-systolic (LVIDs) internal dimensions, and derived fractional shortening (FS) for assessing ventricular systolic function.")

elif st.session_state.nav_tab == "Analyse":
    if not st.session_state.user:
        st.warning("Please sign in or register on the Home page to begin analyses.")
        st.stop()
        
    st.header("Physiological Data Analysis Pipeline")
    modality = st.radio("Modality", ["ECG", "EEG", "Echocardiography"], horizontal=True, key="analyse_modality_radio")

    try:
        if modality == "ECG":
            raw, nm = get_bytes("Upload ECG CSV (time column + signal)", ["csv"], "demo_ecg.csv")
            fs_in = st.number_input("Sampling rate (Hz), only if no time column", 0, 5000, 0, key="ecg_fs_in")
            if raw:
                sig, fs, col = ecg.load_csv(io.BytesIO(raw), fs_in or None)
                res, peaks = ecg.analyse(sig, fs, nm)
                win = st.slider("Window start (s)", 0.0, max(res.quality["duration_s"] - 10, 0.1), 0.0, key="ecg_win")
                a, b = int(win * fs), int((win + 10) * fs)
                t = np.arange(len(sig)) / fs
                fig = go.Figure(go.Scatter(x=t[a:b], y=sig[a:b], line=dict(width=1, color="#0ea5e9"), name=col))
                pk = peaks[(peaks >= a) & (peaks < b)]
                fig.add_trace(go.Scatter(x=t[pk], y=sig[pk], mode="markers", marker=dict(color="#f43f5e", size=7), name="R-peaks"))
                fig.update_layout(height=320, xaxis_title="Time (s)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6"), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                m = res.measurements
                keys = [k for k in ["heart_rate_bpm", "sdnn_ms", "rmssd_ms", "pr_ms", "qrs_ms", "qt_ms"] if k in m]
                cols = st.columns(max(len(keys), 1))
                for c, k in zip(cols, keys):
                    c.metric(k.replace("_", " "), m.get(k, "-"))
                finish(res)

        elif modality == "EEG":
            raw, nm = get_bytes("Upload EEG (EDF/EDF+ or CSV)", ["edf", "csv"], "demo_eeg.csv")
            fs_in = st.number_input("Sampling rate (Hz), only if CSV has no time column", 0, 5000, 0, key="eeg_fs_in")
            if raw:
                data, fs, names = (eeg.load_edf(raw) if nm.lower().endswith(".edf") else eeg.load_csv(io.BytesIO(raw), fs_in or None))
                res = eeg.analyse(data, fs, names, nm)
                win = st.slider("Window start (s)", 0.0, max(res.quality["duration_s"] - 10, 0.1), 0.0, key="eeg_win")
                a, b = int(win * fs), int((win + 10) * fs)
                t = np.arange(data.shape[1]) / fs
                fig = go.Figure()
                for i, c in enumerate(names):
                    seg = data[i, a:b]
                    seg = (seg - np.nanmean(seg)) / (np.nanstd(seg) + 1e-9) if res.measurements["per_channel"][c]["status"] != "flat" else seg * 0
                    fig.add_trace(go.Scatter(x=t[a:b], y=seg * 0.4 - i, line=dict(width=1), name=c))
                fig.update_layout(height=420, showlegend=False, xaxis_title="Time (s)",
                                  yaxis=dict(tickvals=[-i for i in range(len(names))], ticktext=names),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6"),
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                g = res.measurements["global_relative_power"]
                st.plotly_chart(px.bar(x=list(g), y=list(g.values()), labels={"x": "Band", "y": "Relative power"},
                                       title="Global relative band power (usable channels)").update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6")),
                                use_container_width=True)
                st.dataframe(pd.DataFrame(res.measurements["per_channel"]).T)
                finish(res)

        else:
            raw, nm = get_bytes("Upload echo DICOM or image (PNG/JPG)", ["dcm", "png", "jpg", "jpeg"], "demo_echo.dcm")
            if raw:
                frames, meta, mm = (echo.load_dicom(raw) if nm.lower().endswith(".dcm") else echo.load_image(raw))
                if not mm:
                    mm = st.number_input("Calibration (mm per pixel)", 0.0, 5.0, 0.0, 0.01, key="echo_calib") or None
                i = st.slider("Frame", 0, frames.shape[0] - 1, 0, key="echo_frame") if frames.shape[0] > 1 else 0
                st.caption("Place calipers by entering pixel coordinates (x1, y1) to (x2, y2).")
                h, w = frames[i].shape[:2]
                calipers = []
                for lab in ["LVIDd", "LVIDs"]:
                    c = st.columns(5)
                    on = c[0].checkbox(lab, key=f"chk_{lab}")
                    p = [c[k + 1].number_input(n, 0, max(w, h), v, key=f"echo_{lab}_{n}") for k, (n, v) in
                         enumerate([("x1", w // 2 - 50), ("y1", h // 2), ("x2", w // 2 + 50), ("y2", h // 2)])]
                    if on: calipers.append({"label": lab, "p1": (p[0], p[1]), "p2": (p[2], p[3])})
                fig = px.imshow(frames[i], color_continuous_scale="gray", aspect="equal")
                for c in calipers:
                    fig.add_shape(type="line", x0=c["p1"][0], y0=c["p1"][1], x1=c["p2"][0], y1=c["p2"][1],
                                  line=dict(color="red" if c["label"] == "LVIDd" else "cyan", width=3))
                fig.update_layout(coloraxis_showscale=False, height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6"), margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                res = echo.analyse(frames, meta, mm, calipers, nm)
                finish(res)
    except Exception as e:
        db.log_event(st.session_state.user, modality, "failed")
        st.error(f"Could not process this file: {e}")

elif st.session_state.nav_tab == "Teaching Lab":
    st.header("Interactive Teaching Lab & Simulation Sandbox")
    lab_modality = st.radio("Modality Lab", ["ECG", "EEG", "Echocardiography"], horizontal=True, key="lab_mod_select")
    
    if lab_modality == "ECG":
        mode = st.radio("Mode", ["Simulator (show the steps)", "Guided exercises", "Quiz"], horizontal=True, key="ecg_mode")
        FS = 250
        if mode.startswith("Simulator"):
            c = st.columns(5)
            hr = c[0].slider("Heart rate (bpm)", 40, 160, 72, key="sim_hr")
            var = c[1].slider("Variability (bpm SD)", 0.0, 10.0, 1.0, 0.5, key="sim_var")
            noise = c[2].slider("Random noise", 0.0, 0.3, 0.03, 0.01, key="sim_noise")
            wander = c[3].slider("Baseline wander (mV)", 0.0, 0.6, 0.0, 0.05, key="sim_wander")
            mains = c[4].slider("50 Hz interference (mV)", 0.0, 0.4, 0.0, 0.05, key="sim_mains")
            t, sig, truth = simulate_ecg(hr, 30, FS, noise, var, wander, mains)
            res, peaks = ecg.analyse(sig, FS, "simulated")
            st_ = ecg.detect_stages(sig, FS)
            stage = st.radio("Pipeline stage", ["1. Raw signal", "2. Band-pass filtered (5-15 Hz)",
                             "3. Energy and threshold", "4. Detected R-peaks"], horizontal=True, key="ecg_stage")
            a, b = 0, 10 * FS
            if stage.startswith("1"):
                st.plotly_chart(window_plot(t, st_["raw"], a, b, title="Raw signal"), use_container_width=True)
            elif stage.startswith("2"):
                st.plotly_chart(window_plot(t, st_["filtered"], a, b, title="Filtered: QRS emphasised"), use_container_width=True)
            elif stage.startswith("3"):
                st.plotly_chart(window_plot(t, st_["energy"], a, b, thr=st_["threshold"], title="Squared-derivative energy"), use_container_width=True)
            else:
                st.plotly_chart(window_plot(t, st_["raw"], a, b, peaks=peaks, title="R-peaks on original signal"), use_container_width=True)
            m = res.measurements
            cmp = pd.DataFrame({"Simulator setting": [hr, var, "-", "-"],
                                "Measured": [m.get("heart_rate_bpm", "-"), m.get("sdnn_ms", "-"), m.get("rmssd_ms", "-"), res.quality["label"]]},
                               index=["Heart rate (bpm)", "SDNN (ms)", "RMSSD (ms)", "Signal quality"])
            st.table(cmp.astype(str))
        elif mode.startswith("Guided"):
            ex = st.selectbox("Exercise", list(EXERCISES), key="ecg_ex")
            st.markdown(f"**Aim:** {EXERCISES[ex]['aim']}")
            for i, step in enumerate(EXERCISES[ex]["steps"], 1):
                st.markdown(f"{i}. {step}")
            if ex.startswith("Exercise 1"):
                if "ex1" not in st.session_state or st.button("New trace", key="ex1_btn"):
                    rng = np.random.default_rng()
                    st.session_state.ex1 = {"hr": int(rng.integers(50, 111)), "seed": int(rng.integers(1, 10_000)), "shown": False}
                e = st.session_state.ex1
                t, sig, _ = simulate_ecg(e["hr"], 30, FS, 0.03, 1.0, 0, 0, e["seed"])
                st.plotly_chart(window_plot(t, sig, 0, 10 * FS, title="10-second window"), use_container_width=True)
                guess = st.number_input("Your estimate (bpm)", 20, 250, 70, key="ex1_guess")
                if st.button("Check my answer", key="ex1_check"):
                    e["shown"] = True
                if e["shown"]:
                    r, _ = ecg.analyse(sig, FS, "exercise")
                    comp = r.measurements["heart_rate_bpm"]
                    chk = check_manual_hr(guess, comp)
                    st.metric("Algorithm result (bpm)", comp, delta=chk["error_bpm"], delta_color="off")
        else:
            spec = "ECG"
            qz = st.session_state.get("quiz_ecg")
            if qz is None or st.button("New quiz", key="new_quiz_ecg"):
                seed = int(np.random.default_rng().integers(1, 1_000_000))
                st.session_state.quiz_ecg = qz = {"spec": spec, "seed": seed, "qs": quizmod.draw(spec, seed=seed), "result": None}
            with st.form(f"quizform_ecg_{qz['seed']}"):
                picks = []
                for i, q in enumerate(qz["qs"]):
                    picks.append(st.radio(f"**{i + 1}. {q['q']}**", q["options"], index=None, key=f"q_ecg_{qz['seed']}_{i}"))
                submitted = st.form_submit_button("Submit answers")
            if submitted and all(p is not None for p in picks):
                chosen = [q["options"].index(p) for q, p in zip(qz["qs"], picks)]
                qz["result"] = quizmod.score(qz["qs"], chosen)
            if qz["result"]:
                n_ok, rows = qz["result"]
                st.subheader(f"Score: {n_ok} / {len(rows)}")
                for i, r in enumerate(rows, 1):
                    (st.success if r["correct"] else st.error)(f"Q{i}: {'Correct' if r['correct'] else 'Not quite'}")
                    st.info(r["explanation"])

    elif lab_modality == "EEG":
        mode = st.radio("Mode", ["Simulator (show the steps)", "Quiz"], horizontal=True, key="eeg_mode_select")
        if mode.startswith("Simulator"):
            c = st.columns(3)
            state = c[0].radio("Subject State", ["Eyes closed", "Eyes open"], horizontal=True, key="eeg_state")
            alpha_mult = c[1].slider("Alpha Amplitude Multiplier", 0.2, 3.0, 1.0, 0.2, key="eeg_alpha_slider")
            noise_lvl = c[2].slider("Artifact / Muscle Noise", 0.0, 0.5, 0.05, 0.05, key="eeg_noise_slider")
            sim_res = simulate_eeg_pipeline(state=state, alpha_amp_mult=alpha_mult, noise_level=noise_lvl)
            t, raw, filtered, energy, ch_names = sim_res["t"], sim_res["raw"], sim_res["filtered"], sim_res["energy"], sim_res["ch_names"]
            fig = go.Figure()
            for i, c_name in enumerate(ch_names):
                fig.add_trace(go.Scatter(x=t[:2500], y=raw[i, :2500] - (i * 30), mode="lines", name=c_name))
            fig.update_layout(height=380, title=f"Raw Multi-Channel EEG Traces ({state})", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f3f4f6"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            spec = "EEG"
            qz = st.session_state.get("quiz_eeg")
            if qz is None or st.button("New quiz", key="new_quiz_eeg"):
                seed = int(np.random.default_rng().integers(1, 1_000_000))
                st.session_state.quiz_eeg = qz = {"spec": spec, "seed": seed, "qs": quizmod.draw(spec, seed=seed), "result": None}
            with st.form(f"quizform_eeg_{qz['seed']}"):
                picks = [st.radio(f"**{i + 1}. {q['q']}**", q["options"], index=None, key=f"q_eeg_{qz['seed']}_{i}") for i, q in enumerate(qz["qs"])]
                if st.form_submit_button("Submit answers") and all(p is not None for p in picks):
                    qz["result"] = quizmod.score(qz["qs"], [q["options"].index(p) for q, p in zip(qz["qs"], picks)])
            if qz["result"]:
                st.subheader(f"Score: {qz['result'][0]} / {len(qz['result'][1])}")

    else:
        mode = st.radio("Mode", ["Simulator (show the steps)", "Quiz"], horizontal=True, key="echo_mode_select")
        if mode.startswith("Simulator"):
            c = st.columns(3)
            lvid_d = c[0].slider("LVIDd (mm)", 35.0, 70.0, 50.0, key="sim_lvid_d")
            lvid_s = c[1].slider("LVIDs (mm)", 20.0, 55.0, 30.0, key="sim_lvid_s")
            hr_sim = c[2].slider("Heart Rate (bpm)", 50, 120, 72, key="sim_hr_echo")
            echo_res = simulate_echo_pipeline(lvid_d=lvid_d, lvid_s=lvid_s, heart_rate=hr_sim)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Fractional Shortening", f"{echo_res['fractional_shortening_pct']}%")
            sc2.metric("Ejection Fraction", f"{echo_res['ejection_fraction_pct']}%")
            sc3.metric("Cardiac Output", f"{echo_res['cardiac_output_lmin']} L/min")
        else:
            spec = "Echocardiography"
            qz = st.session_state.get("quiz_echo")
            if qz is None or st.button("New quiz", key="new_quiz_echo"):
                seed = int(np.random.default_rng().integers(1, 1_000_000))
                st.session_state.quiz_echo = qz = {"spec": spec, "seed": seed, "qs": quizmod.draw(spec, seed=seed), "result": None}
            with st.form(f"quizform_echo_{qz['seed']}"):
                picks = [st.radio(f"**{i + 1}. {q['q']}**", q["options"], index=None, key=f"q_echo_{qz['seed']}_{i}") for i, q in enumerate(qz["qs"])]
                if st.form_submit_button("Submit answers") and all(p is not None for p in picks):
                    qz["result"] = quizmod.score(qz["qs"], [q["options"].index(p) for q, p in zip(qz["qs"], picks)])
            if qz["result"]:
                st.subheader(f"Score: {qz['result'][0]} / {len(qz['result'][1])}")

elif st.session_state.nav_tab == "About":
    st.markdown("""
    ### About PhysioLab Intelligence
    **Architecture:** Modular scientific engines decoupled from Streamlit UI (`physiolab/`).  
    **Pedagogical Goal:** Trusted by university physiology departments for interactive signal processing, automated practical grading, and transparent multi-stage algorithmic breakdowns.
    """)

# ---------- Edge-to-Edge Absolute Bottom Footer ----------
st.markdown("""
<div class="footer-container">
    <div class="footer-brand">🫀 PHYSIOLAB INTELLIGENCE · <a href="https://www.fsi-aibr.org" target="_blank" style="color: #0ea5e9; text-decoration: none;">FSI-AIBR</a></div>
    <p style="margin: 0; color: #94a3b8;">
        Developed by <a href="mailto:da.zx@outlook.com" style="color: #ffffff; text-decoration: underline;"><b>Daniel Onimisi</b></a> | Secure In-Memory Physiological Intelligence Platform
    </p>
    <p style="margin-top: 0.4rem; font-size: 0.75rem; color: #64748b;">
        © 2026 <a href="https://www.fsi-aibr.org" target="_blank" style="color: #64748b; text-decoration: none;">FSI-AIBR</a>. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)