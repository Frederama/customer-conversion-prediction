import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Conversion Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.main, .stApp { background-color: #f7f6f2; }

.hero {
    background: #0f1923;
    color: #f7f6f2;
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.4rem; margin-bottom: 0.3rem; color: #f7f6f2; }
.hero p  { color: #8a9ba8; font-size: 1rem; font-weight: 300; margin: 0; }

.result-box {
    background: #0f1923;
    color: #f7f6f2;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-box .prob    { font-family: 'DM Serif Display', serif; font-size: 4rem; line-height: 1; }
.result-box .verdict { font-size: 1rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.5rem; }
.high { color: #4ade80; }
.low  { color: #f87171; }

.section-label {
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7280;
    margin-bottom: 0.75rem; margin-top: 1.5rem;
}

.insight-card {
    background: #ffffff; border-radius: 10px;
    padding: 1.2rem 1.5rem; margin-bottom: 0.75rem;
    border-left: 4px solid #0f1923;
}
.insight-card p { margin: 0; font-size: 0.9rem; color: #374151; }
.insight-card strong { color: #0f1923; }

div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }

.stButton > button {
    background-color: #0f1923; color: #f7f6f2;
    border: none; border-radius: 8px;
    padding: 0.65rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 0.95rem;
    width: 100%; letter-spacing: 0.03em;
}
.stButton > button:hover { background-color: #1e3a4f; }

/* Force light mode on all selectbox elements */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] > div > div > div {
    background-color: #ffffff !important;
    color: #0f1923 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] span { color: #0f1923 !important; }

/* Fix all input labels */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #374151 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Slider value text */
div[data-testid="stSlider"] p { color: #374151 !important; }

/* Section label contrast */
.section-label { color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("best_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, threshold, explainer, label_encoders

model, feature_cols, threshold, explainer, label_encoders = load_artifacts()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Customer Conversion Predictor</h1>
    <p>Predicts the likelihood a customer will subscribe to a term deposit —
       helping teams prioritize outreach and reduce campaign costs.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_inputs, col_results = st.columns([1, 1.3], gap="large")

# ── LEFT: Inputs ───────────────────────────────────────────────────────────────
with col_inputs:
    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

    age = st.slider("Age", 18, 95, 40)
    job = st.selectbox("Job Type", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    marital   = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education", [
        "basic.4y", "basic.6y", "basic.9y", "high.school",
        "illiterate", "professional.course", "university.degree", "unknown"
    ])
    default = st.selectbox("Credit in Default?", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing Loan?",      ["no", "yes", "unknown"])
    loan    = st.selectbox("Personal Loan?",     ["no", "yes", "unknown"])

    st.markdown('<div class="section-label">Last Contact</div>', unsafe_allow_html=True)

    contact     = st.selectbox("Contact Type", ["cellular", "telephone"])
    month       = st.selectbox("Month of Last Contact", [
        "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
    ])
    day_of_week = st.selectbox("Day of Week", ["mon","tue","wed","thu","fri"])
    duration    = st.slider("Call Duration (seconds)", 0, 5000, 250,
                            help="Note: duration is only known after the call ends.")

    st.markdown('<div class="section-label">Campaign History</div>', unsafe_allow_html=True)

    campaign = st.slider("Contacts This Campaign", 1, 50, 2)
    pdays    = st.slider("Days Since Last Contact (0 = never contacted)", 0, 30, 0)
    previous = st.slider("Previous Campaign Contacts", 0, 10, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "nonexistent", "success"])

    st.markdown('<div class="section-label">Economic Indicators</div>', unsafe_allow_html=True)

    euribor3m     = st.slider("Euribor 3-Month Rate", 0.5, 5.5, 3.5, step=0.01)
    cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.5, step=0.01)
    cons_conf_idx  = st.slider("Consumer Confidence Index", -51.0, -26.0, -40.0, step=0.1)

    predict_btn = st.button("Predict Conversion Likelihood")

# ── RIGHT: Results ─────────────────────────────────────────────────────────────
with col_results:

    if predict_btn:

        # Build zero-filled input row aligned to training features
        input_data = {col: 0 for col in feature_cols}

        # Numeric
        input_data["age"]            = age
        input_data["duration"]       = duration
        input_data["campaign"]       = campaign
        input_data["pdays"]          = pdays
        input_data["previous"]       = previous
        input_data["euribor3m"]      = euribor3m
        input_data["cons.price.idx"] = cons_price_idx
        input_data["cons.conf.idx"]  = cons_conf_idx

        # Binary label-encoded using saved LabelEncoders — guaranteed to match training
        input_data["default"] = int(label_encoders["default"].transform([default])[0])
        input_data["housing"] = int(label_encoders["housing"].transform([housing])[0])
        input_data["loan"]    = int(label_encoders["loan"].transform([loan])[0])
        input_data["contact"] = int(label_encoders["contact"].transform([contact])[0])

        # One-hot encoded
        for key, val in {
            f"job_{job}": 1,
            f"marital_{marital}": 1,
            f"education_{education}": 1,
            f"month_{month}": 1,
            f"day_of_week_{day_of_week}": 1,
            f"poutcome_{poutcome}": 1,
        }.items():
            if key in input_data:
                input_data[key] = val

        input_df = pd.DataFrame([input_data])

        # Predict
        prob    = model.predict_proba(input_df)[0][1]
        verdict = prob >= threshold
        color_class  = "high" if verdict else "low"
        verdict_text = "Likely to Convert" if verdict else "Unlikely to Convert"

        # Result box
        st.markdown(f"""
        <div class="result-box">
            <div class="prob {color_class}">{prob:.1%}</div>
            <div class="verdict" style="color:{'#4ade80' if verdict else '#f87171'}">
                {verdict_text}
            </div>
            <div style="color:#8a9ba8; font-size:0.8rem; margin-top:0.75rem;">
                Decision threshold: {threshold:.2f} &nbsp;·&nbsp; Model: LightGBM
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Business insight cards
        st.markdown('<div class="section-label">Key Drivers</div>', unsafe_allow_html=True)

        if duration > 400:
            st.markdown('<div class="insight-card"><p>📞 <strong>Long call duration</strong> — engaged customers are significantly more likely to convert.</p></div>', unsafe_allow_html=True)
        if month in ["mar", "sep", "oct", "dec"]:
            st.markdown('<div class="insight-card"><p>📅 <strong>Favourable contact month</strong> — these months show historically higher subscription rates.</p></div>', unsafe_allow_html=True)
        if poutcome == "success":
            st.markdown('<div class="insight-card"><p>✅ <strong>Previous campaign success</strong> is a strong positive signal for conversion.</p></div>', unsafe_allow_html=True)
        if euribor3m < 2.0:
            st.markdown('<div class="insight-card"><p>📉 <strong>Low Euribor rate</strong> — customers tend to subscribe more in low interest-rate environments.</p></div>', unsafe_allow_html=True)
        if duration < 100:
            st.markdown('<div class="insight-card"><p>⚠️ <strong>Very short call</strong> — brief calls rarely convert. Consider a re-engagement strategy.</p></div>', unsafe_allow_html=True)
        if campaign > 10:
            st.markdown('<div class="insight-card"><p>⚠️ <strong>High contact frequency</strong> — too many contacts in a campaign can reduce conversion likelihood.</p></div>', unsafe_allow_html=True)

        # SHAP waterfall
        st.markdown('<div class="section-label">SHAP Explanation — Why This Prediction?</div>', unsafe_allow_html=True)

        shap_values = explainer.shap_values(input_df)
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        base_val  = explainer.expected_value
        base_val  = base_val[1] if isinstance(base_val, list) else base_val

        shap_explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=base_val,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )

        fig, _ = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#ffffff")
        shap.waterfall_plot(shap_explanation, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <p style="font-size:0.78rem; color:#9ca3af; margin-top:0.5rem;">
        🔴 Red bars push the prediction toward conversion. 🔵 Blue bars push it away.
        Bar length indicates how much each feature influenced this specific prediction.
        </p>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#ffffff; border-radius:12px; padding:3rem 2rem;
                    text-align:center; border:1px dashed #d1d5db; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📊</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.4rem;
                        color:#0f1923; margin-bottom:0.5rem;">Ready to predict</div>
            <div style="color:#9ca3af; font-size:0.9rem;">
                Fill in the customer details on the left and click Predict.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none; border-top:1px solid #e5e7eb; margin-top:3rem;">
<p style="text-align:center; color:#9ca3af; font-size:0.78rem;">
Built with LightGBM · SHAP · Streamlit &nbsp;|&nbsp; Bank Marketing Dataset (UCI)
&nbsp;|&nbsp; ⚠️ Call duration is post-call only — for pre-call scoring, retrain without this feature.
</p>
""", unsafe_allow_html=True)
