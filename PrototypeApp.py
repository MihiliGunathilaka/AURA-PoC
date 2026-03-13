import streamlit as st
import pandas as pd
import requests
from email_actions import render_email_draft_popup


# CONFIG

API_URL = "aura-api-production-21e5.up.railway.app/predict"

st.set_page_config(
    page_title="AURA – Churn Risk & Lead-Time Prototype",
    layout="wide"
)


# TITLE & INTRO


st.title("AURA – Cloud User Churn Risk & Lead-Time Prototype")

st.write("""
This prototype demonstrates **real-time churn prediction** using a **dual-model architecture**:

**Model 1:** XGBoost multi-class classifier → *Safe / Warning / At-Risk*  
**Model 2:** Survival-based model → *Estimated time until churn (days)*  

Predictions are served via **FastAPI** and visualised here in **Streamlit**.
""")


# LOAD RAW DATA

@st.cache_data
def load_data():
    df = pd.read_csv("AURA_cloud_user_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
st.success("Raw behavioural dataset loaded")


# SIDEBAR – USER SELECTION


st.sidebar.header("User Selection")

user_ids = sorted(df["user_id"].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

df_user = df[df["user_id"] == selected_user].sort_values("date")

if df_user.empty:
    st.warning("No data available for this user.")
    st.stop()

latest = df_user.iloc[-1]


# SNAPSHOT METRICS


st.subheader("Latest Behaviour Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Logins", int(latest["login_count"]))
c2.metric("Session (min)", round(latest["session_duration_minutes"], 1))
c3.metric("API Calls", int(latest["api_calls"]))
c4.metric("CPU Usage (%)", round(latest["cpu_usage_percent"], 1))


# BEHAVIOUR TRENDS


st.subheader("Behaviour Trends")

st.line_chart(df_user.set_index("date")[["login_count"]], height=200)
st.line_chart(df_user.set_index("date")[["session_duration_minutes"]], height=200)


# ADDITIONAL TREND EXPLORER


st.subheader("Additional Behavioural Trend Explorer")

trend_metrics = {
    "API Calls": "api_calls",
    "Feature Usage Score": "feature_usage_score",
    "Resource Consumption (MB)": "resource_consumption_mb",
    "CPU Usage (%)": "cpu_usage_percent",
    "Engagement Score": "engagement_score"
}

selected_metric_label = st.selectbox(
    "Select a behavioural metric:",
    list(trend_metrics.keys())
)

selected_metric = trend_metrics[selected_metric_label]

st.line_chart(
    df_user.set_index("date")[[selected_metric]],
    height=250
)


# FEATURE CALCULATIONS (MATCH MODEL TRAINING)


login_avg_7d = df_user["login_count"].tail(7).mean()
session_avg_7d = df_user["session_duration_minutes"].tail(7).mean()

login_trend_7d = (
    df_user["login_count"].iloc[-1] - df_user["login_count"].iloc[-8]
    if len(df_user) >= 8 else 0
)

engagement_score = (
    latest["login_count"] * 0.3
    + latest["session_duration_minutes"] * 0.2
    + latest["api_calls"] * 0.3
    + latest["resource_consumption_mb"] * 0.2
)

resource_efficiency = (
    latest["api_calls"] / (latest["resource_consumption_mb"] + 1e-3)
)


# BUILD API PAYLOAD


payload = {
    "region": latest["region"],
    "subscription_tier": latest["subscription_tier"],
    "tenure_days": float(latest["tenure_days"]),

    "login_count": float(latest["login_count"]),
    "session_duration_minutes": float(latest["session_duration_minutes"]),
    "api_calls": float(latest["api_calls"]),
    "feature_usage_score": float(latest["feature_usage_score"]),
    "resource_consumption_mb": float(latest["resource_consumption_mb"]),
    "cpu_usage_percent": float(latest["cpu_usage_percent"]),
    "support_tickets_created": float(latest["support_tickets_created"]),

    "login_avg_7d": float(login_avg_7d),
    "session_avg_7d": float(session_avg_7d),
    "login_trend_7d": float(login_trend_7d),
    "engagement_score": float(engagement_score),

    "inactivity_days": float(latest.get("inactivity_days", 0)),
    "usage_drop_pct": float(latest.get("usage_drop_pct", 0)),
    "volatility_score": float(latest.get("volatility_score", 0)),
    "resource_efficiency": float(resource_efficiency),
}


# CALL FASTAPI


st.subheader("ML Model Prediction")

try:
    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    result = response.json()
except Exception as e:
    st.error("Prediction API is not available")
    st.code(str(e))
    st.stop()


# PARSE RESPONSE (UPDATED)


predicted_state = result["risk_state"]
probabilities = result["risk_probabilities"]
lead_time = result.get("lead_time_days")
ci_low = result.get("confidence_interval_lower")
ci_high = result.get("confidence_interval_upper")


# DISPLAY RISK RESULT


if predicted_state == "At-Risk":
    st.error(f"Predicted Risk State: **{predicted_state}**")
elif predicted_state == "Warning":
    st.warning(f"Predicted Risk State: **{predicted_state}**")
else:
    st.success(f"Predicted Risk State: **{predicted_state}**")

st.write("### Risk Probabilities")
st.json(probabilities)


# LEAD-TIME TO CHURN SECTION (MODEL 2)


st.subheader(" Predicted Lead Time to Churn")

if lead_time is not None:
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimated Time to Churn",
        f"{lead_time:.1f} days"
    )

    c2.metric(
        "Lower Bound (Conservative)",
        f"{ci_low:.1f} days"
    )

    c3.metric(
        "Upper Bound",
        f"{ci_high:.1f} days"
    )

    # Interpretation
    if lead_time <= 7:
        st.error(" **Critical:** Immediate churn risk (≤ 7 days)")
    elif lead_time <= 30:
        st.warning(" **Urgent:** Churn likely within 30 days")
    else:
        st.success(" **Stable:** No immediate churn expected")

else:
    st.info(
        "Lead-time prediction is not applicable for low-risk users "
        "(Safe users are treated as right-censored)."
    )


# ADAPTIVE ENGAGEMENT


st.subheader("AURA Recommended Action")

if predicted_state == "At-Risk":
    st.write("""
    **Immediate Intervention**
    - Personalised re-engagement email  
    - Discount or free trial  
    - Support follow-up  
    """)
elif predicted_state == "Warning":
    st.write("""
    **Preventive Engagement**
    - Feature nudges  
    - In-app reminders  
    - Short tutorials  
    """)
else:
    st.write("""
    **Maintain Engagement**
    - Regular newsletters  
    - Encourage feature exploration  
    """)


# EMAIL ACTION


if predicted_state in ["Warning", "At-Risk"]:
    if "show_email_popup" not in st.session_state:
        st.session_state.show_email_popup = False

    if st.button("Send Engagement Email"):
        st.session_state.show_email_popup = True

    if st.session_state.show_email_popup:
        render_email_draft_popup(
            predicted_state=predicted_state,
            user_id=str(selected_user),
            to_email_default="support@auracloud.example"
        )


# DEBUG


with st.expander("Debug: Payload sent to API"):
    st.json(payload)
