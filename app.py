"""
Telco Customer Churn Prediction Dashboard
Production-ready Streamlit app with ML model, analytics, and explainability.
"""

import os
import io
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* KPI cards */
.kpi-card {
    background: #f8faff;
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 5px solid #4361ee;
    margin-bottom: 8px;
}
.kpi-label { font-size: 12px; font-weight: 600; color: #6c757d; letter-spacing: .6px; text-transform: uppercase; }
.kpi-value { font-size: 28px; font-weight: 700; color: #1d3557; }
.kpi-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }

/* Section headers */
.sec-header {
    font-size: 17px; font-weight: 700; color: #1d3557;
    border-bottom: 2px solid #4361ee; padding-bottom: 6px; margin: 20px 0 12px 0;
}

/* Prediction result boxes */
.result-churn    { background:#fff0f0; border:2px solid #e63946; border-radius:12px; padding:20px; text-align:center; }
.result-nochurn  { background:#f0fff4; border:2px solid #2dc653; border-radius:12px; padding:20px; text-align:center; }
.result-title    { font-size:22px; font-weight:700; margin-bottom:6px; }
.result-prob     { font-size:15px; color:#555; }

/* Sidebar */
div[data-testid="stSidebar"] { background: #1d3557; }
div[data-testid="stSidebar"] * { color: #e0e8ff !important; }

/* Metric containers */
div[data-testid="metric-container"] {
    background: #f0f4ff; border-radius: 10px;
    padding: 0.4rem 0.8rem; border: 1px solid #dce3f3;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

CAT_OPTIONS = {
    "gender":           ["Female", "Male"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["No", "Yes"],
    "PhoneService":     ["No", "Yes"],
    "MultipleLines":    ["No phone service", "No", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_raw_data() -> pd.DataFrame:
    """Load and lightly clean the raw CSV."""
    candidates = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].replace({" ": "0"}), errors="coerce"
            ).fillna(0.0)
            return df
    st.error("❌ CSV file not found. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the app folder.")
    st.stop()


@st.cache_resource(show_spinner="Training model…")
def train_model(df: pd.DataFrame):
    """
    Preprocess data and train a RandomForest classifier.
    Returns (model, encoders, X_test, y_test, feature_names).
    """
    data = df.drop(columns=["customerID"]).copy()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Label-encode all object columns
    encoders: dict[str, LabelEncoder] = {}
    obj_cols = data.select_dtypes(include="object").columns
    for col in obj_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, encoders, X_test, y_test, X.columns.tolist()


def predict_single(model, encoders: dict, input_dict: dict) -> tuple[int, float]:
    """Encode one input record and return (class, churn_probability)."""
    row = pd.DataFrame([input_dict])
    for col, enc in encoders.items():
        if col in row.columns:
            val = row[col].astype(str)
            # Handle unseen labels gracefully
            known = set(enc.classes_)
            row[col] = val.apply(lambda v: enc.transform([v])[0] if v in known else 0)
    prob = model.predict_proba(row)[0][1]
    pred = int(prob >= 0.5)
    return pred, float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔮 Predict Churn", "📊 Insights & Analytics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>Telco Customer Churn · RandomForest · Streamlit</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
raw_df = load_raw_data()
model, encoders, X_test, y_test, feature_cols = train_model(raw_df)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
acc          = accuracy_score(y_test, y_pred)
report_dict  = classification_report(y_test, y_pred, output_dict=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("📡 Telco Customer Churn Dashboard")
    st.markdown("**End-to-end ML pipeline** — RandomForest classifier trained on 7 043 customers.")

    # ── KPI row ──────────────────────────────────────────────────────────────
    total         = len(raw_df)
    churned       = (raw_df["Churn"] == "Yes").sum()
    churn_rate    = churned / total * 100
    avg_monthly   = raw_df["MonthlyCharges"].mean()
    avg_tenure    = raw_df["tenure"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("👥 Total Customers",   f"{total:,}")
    k2.metric("⚠️ Churned",            f"{churned:,}")
    k3.metric("📉 Churn Rate",         f"{churn_rate:.1f}%")
    k4.metric("💵 Avg Monthly Bill",   f"${avg_monthly:.2f}")
    k5.metric("📅 Avg Tenure (mo)",    f"{avg_tenure:.1f}")

    st.markdown("---")

    # ── Model performance ────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Model Performance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc*100:.1f}%")
    m2.metric("Precision (Churn)", f"{report_dict['1']['precision']*100:.1f}%")
    m3.metric("Recall (Churn)",    f"{report_dict['1']['recall']*100:.1f}%")
    m4.metric("F1 Score (Churn)",  f"{report_dict['1']['f1-score']*100:.1f}%")

    c1, c2 = st.columns(2)

    # Confusion matrix
    with c1:
        st.markdown('<div class="sec-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual"},
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
        )
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    with c2:
        st.markdown('<div class="sec-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {roc_auc:.3f}",
                                     line=dict(color="#4361ee", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="gray", width=1),
                                     showlegend=False))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=320, margin=dict(t=10, b=10), legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    st.markdown('<div class="sec-header">Feature Importance</div>', unsafe_allow_html=True)
    fi = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=True)
    )
    fig_fi = px.bar(
        fi, orientation="h",
        color=fi.values,
        color_continuous_scale="Blues",
        labels={"value": "Importance", "index": "Feature"},
    )
    fig_fi.update_layout(
        height=500, margin=dict(t=10, b=10),
        coloraxis_showscale=False, showlegend=False
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Dataset preview
    st.markdown('<div class="sec-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════
# PAGE 2 — PREDICT CHURN
# ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict Churn":
    st.title("🔮 Customer Churn Predictor")
    st.markdown("Fill in the customer profile below and get an instant churn prediction.")

    with st.form("prediction_form"):
        st.markdown('<div class="sec-header">Demographics</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        gender        = d1.selectbox("Gender",       CAT_OPTIONS["gender"])
        senior        = d2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner       = d3.selectbox("Partner",       CAT_OPTIONS["Partner"])
        dependents    = d4.selectbox("Dependents",    CAT_OPTIONS["Dependents"])

        st.markdown('<div class="sec-header">Account Info</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        tenure           = a1.slider("Tenure (months)", 0, 72, 12)
        contract         = a2.selectbox("Contract Type",    CAT_OPTIONS["Contract"])
        paperless        = a3.selectbox("Paperless Billing", CAT_OPTIONS["PaperlessBilling"])

        a4, a5 = st.columns(2)
        payment_method   = a4.selectbox("Payment Method",   CAT_OPTIONS["PaymentMethod"])
        monthly_charges  = a5.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges    = st.number_input(
            "Total Charges ($)", 0.0, 10000.0,
            float(monthly_charges * max(tenure, 1)), step=1.0
        )

        st.markdown('<div class="sec-header">Services</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        phone_service    = s1.selectbox("Phone Service",      CAT_OPTIONS["PhoneService"])
        multiple_lines   = s2.selectbox("Multiple Lines",     CAT_OPTIONS["MultipleLines"])
        internet_service = s3.selectbox("Internet Service",   CAT_OPTIONS["InternetService"])

        s4, s5, s6 = st.columns(3)
        online_security  = s4.selectbox("Online Security",    CAT_OPTIONS["OnlineSecurity"])
        online_backup    = s5.selectbox("Online Backup",      CAT_OPTIONS["OnlineBackup"])
        device_protect   = s6.selectbox("Device Protection",  CAT_OPTIONS["DeviceProtection"])

        s7, s8, s9 = st.columns(3)
        tech_support     = s7.selectbox("Tech Support",       CAT_OPTIONS["TechSupport"])
        streaming_tv     = s8.selectbox("Streaming TV",       CAT_OPTIONS["StreamingTV"])
        streaming_movies = s9.selectbox("Streaming Movies",   CAT_OPTIONS["StreamingMovies"])

        submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)

    if submitted:
        input_dict = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protect, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        pred, prob = predict_single(model, encoders, input_dict)
        churn_pct  = prob * 100
        safe_pct   = (1 - prob) * 100

        # ── Result card ──────────────────────────────────────────────────────
        st.markdown("---")
        if pred == 1:
            st.markdown(
                f'<div class="result-churn">'
                f'<div class="result-title" style="color:#e63946">⚠️ HIGH CHURN RISK</div>'
                f'<div class="result-prob">Churn probability: <strong>{churn_pct:.1f}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-nochurn">'
                f'<div class="result-title" style="color:#2dc653">✅ LOW CHURN RISK</div>'
                f'<div class="result-prob">Retention probability: <strong>{safe_pct:.1f}%</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Gauge chart ──────────────────────────────────────────────────────
        st.markdown("")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_pct,
            delta={"reference": 26.5, "valueformat": ".1f", "suffix": "%"},
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn Probability", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#e63946" if pred == 1 else "#2dc653"},
                "steps": [
                    {"range": [0, 30],  "color": "#d8f5e1"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100],"color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "#1d3557", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Probability bar ──────────────────────────────────────────────────
        fig_bar = go.Figure(go.Bar(
            x=[safe_pct, churn_pct],
            y=["No Churn", "Churn"],
            orientation="h",
            marker_color=["#2dc653", "#e63946"],
            text=[f"{safe_pct:.1f}%", f"{churn_pct:.1f}%"],
            textposition="inside",
        ))
        fig_bar.update_layout(
            height=160, margin=dict(t=0, b=0), showlegend=False,
            xaxis=dict(range=[0, 100], showticklabels=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Feature Importance for this prediction ───────────────────────────
        st.markdown('<div class="sec-header">Key Drivers (Global Feature Importance)</div>',
                    unsafe_allow_html=True)
        fi = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False).head(10)
        )
        fig_fi2 = px.bar(
            fi[::-1], orientation="h",
            color=fi[::-1].values, color_continuous_scale="Reds",
            labels={"value": "Importance", "index": ""},
        )
        fig_fi2.update_layout(
            height=320, margin=dict(t=10, b=10), coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi2, use_container_width=True)

        # ── Download ─────────────────────────────────────────────────────────
        result_df = pd.DataFrame([{
            **input_dict,
            "Prediction": "Churn" if pred == 1 else "No Churn",
            "Churn_Probability_%": round(churn_pct, 2),
        }])
        st.download_button(
            "⬇️ Download Prediction (CSV)",
            data=result_df.to_csv(index=False).encode(),
            file_name="churn_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════
# PAGE 3 — INSIGHTS & ANALYTICS
# ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
else:  # Insights & Analytics
    st.title("📊 Insights & Analytics")
    st.markdown("Explore churn patterns across customer segments.")

    df = raw_df.copy()
    df["ChurnBinary"] = (df["Churn"] == "Yes").astype(int)

    # ── Churn distribution ───────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Churn Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig_pie = px.pie(counts, names="Churn", values="Count", hole=0.55,
                         color_discrete_sequence=["#2dc653", "#e63946"])
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_hist = px.histogram(df, x="tenure", color="Churn", barmode="overlay",
                                color_discrete_map={"Yes": "#e63946", "No": "#4361ee"},
                                labels={"tenure": "Tenure (months)"},
                                opacity=0.75)
        fig_hist.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Charges analysis ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Monthly Charges by Churn Status</div>',
                unsafe_allow_html=True)
    fig_box = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                     color_discrete_map={"Yes": "#e63946", "No": "#4361ee"},
                     points="outliers")
    fig_box.update_layout(height=320, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Contract & Internet ──────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Churn Rate by Segment</div>',
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)

    with s1:
        seg = df.groupby("Contract")["ChurnBinary"].mean().mul(100).reset_index()
        seg.columns = ["Contract", "Churn Rate %"]
        fig_con = px.bar(seg, x="Contract", y="Churn Rate %",
                         color="Churn Rate %", color_continuous_scale="Reds",
                         text=seg["Churn Rate %"].apply(lambda x: f"{x:.1f}%"))
        fig_con.update_traces(textposition="outside")
        fig_con.update_layout(height=300, margin=dict(t=10, b=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_con, use_container_width=True)

    with s2:
        seg2 = df.groupby("InternetService")["ChurnBinary"].mean().mul(100).reset_index()
        seg2.columns = ["Internet Service", "Churn Rate %"]
        fig_int = px.bar(seg2, x="Internet Service", y="Churn Rate %",
                         color="Churn Rate %", color_continuous_scale="Oranges",
                         text=seg2["Churn Rate %"].apply(lambda x: f"{x:.1f}%"))
        fig_int.update_traces(textposition="outside")
        fig_int.update_layout(height=300, margin=dict(t=10, b=10),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_int, use_container_width=True)

    # ── Payment method ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Churn Rate by Payment Method</div>',
                unsafe_allow_html=True)
    pay = df.groupby("PaymentMethod")["ChurnBinary"].mean().mul(100).sort_values(ascending=False).reset_index()
    pay.columns = ["Payment Method", "Churn Rate %"]
    fig_pay = px.bar(pay, x="Churn Rate %", y="Payment Method", orientation="h",
                     color="Churn Rate %", color_continuous_scale="Purples",
                     text=pay["Churn Rate %"].apply(lambda x: f"{x:.1f}%"))
    fig_pay.update_traces(textposition="outside")
    fig_pay.update_layout(height=300, margin=dict(t=10, b=10),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_pay, use_container_width=True)

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Correlation Heatmap (Numeric Features)</div>',
                unsafe_allow_html=True)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "ChurnBinary", "SeniorCitizen"]
    corr = df[num_cols].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        labels={"color": "Correlation"},
    )
    fig_heat.update_layout(height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Gender & Senior ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">Demographics vs Churn</div>',
                unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        gender_churn = df.groupby("gender")["ChurnBinary"].mean().mul(100).reset_index()
        gender_churn.columns = ["Gender", "Churn Rate %"]
        fig_g = px.bar(gender_churn, x="Gender", y="Churn Rate %",
                       color="Gender", color_discrete_sequence=["#4361ee", "#e63946"],
                       text=gender_churn["Churn Rate %"].apply(lambda x: f"{x:.1f}%"))
        fig_g.update_traces(textposition="outside")
        fig_g.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_g, use_container_width=True)

    with g2:
        sr = df.groupby("SeniorCitizen")["ChurnBinary"].mean().mul(100).reset_index()
        sr["SeniorCitizen"] = sr["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        sr.columns = ["Segment", "Churn Rate %"]
        fig_sr = px.bar(sr, x="Segment", y="Churn Rate %",
                        color="Churn Rate %", color_continuous_scale="Reds",
                        text=sr["Churn Rate %"].apply(lambda x: f"{x:.1f}%"))
        fig_sr.update_traces(textposition="outside")
        fig_sr.update_layout(height=300, margin=dict(t=10, b=10),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_sr, use_container_width=True)

    # ── Download full dataset ────────────────────────────────────────────────
    st.markdown("---")
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download Full Dataset with Churn Flag",
        data=buf.getvalue(),
        file_name="churn_data_enriched.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr><center><small>Customer Churn Dashboard · RandomForest · Streamlit</small></center>",
    unsafe_allow_html=True,
)
