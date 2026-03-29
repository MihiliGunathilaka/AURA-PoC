import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import matplotlib.pyplot as plt
from email_actions import render_email_draft_popup


# CONFIG

API_URL = "https://aura-api-production-21e5.up.railway.app/predict"

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
    df = pd.read_csv("AURA_poc_dataset_100users.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()
st.success("Raw behavioural dataset loaded")

# LOAD SHAP EXPLAINER

@st.cache_resource
def load_shap_explainer():
    try:
        explainer = joblib.load("AURA_shap_explainer.pkl")
        model = joblib.load("AURA_xgboost_model.pkl")
        with open("AURA_feature_names.txt") as f:
            feature_names = [l.strip() for l in f if l.strip()]
        return explainer, model, feature_names
    except Exception as e:
        st.warning(f"SHAP explainer not available: {e}")
        return None, None, None

shap_explainer, local_model, shap_feature_names = load_shap_explainer()

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

# SHAP FEATURE IMPORTANCE (MODEL EXPLAINABILITY)

st.subheader("Why This Prediction? — Feature Importance (SHAP)")

if shap_explainer is not None and local_model is not None and shap_feature_names is not None:
    try:
        # Build the same feature vector used for Model 1 prediction
        snapshot_data = payload.copy()
        row = []
        for feat in shap_feature_names:
            if feat.startswith("region_"):
                row.append(1.0 if snapshot_data["region"] == feat.replace("region_", "") else 0.0)
            elif feat.startswith("subscription_tier_"):
                row.append(1.0 if snapshot_data["subscription_tier"] == feat.replace("subscription_tier_", "") else 0.0)
            else:
                row.append(float(snapshot_data.get(feat, 0.0)))

        X_shap = pd.DataFrame([row], columns=shap_feature_names)

        # Get SHAP values
        shap_values = shap_explainer.shap_values(X_shap)

        # Get the predicted class index
        predicted_class_idx = {"Safe": 0, "Warning": 1, "At-Risk": 2}[predicted_state]

        # Get SHAP values for the predicted class
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            class_shap = np.array(shap_values[predicted_class_idx]).flatten()
        elif shap_values.ndim == 3:
            # Shape: (1, num_features, num_classes)
            class_shap = shap_values[0, :, predicted_class_idx].flatten()
        elif shap_values.ndim == 2:
            # Shape: (1, num_features)
            class_shap = shap_values[0].flatten()
        else:
            class_shap = np.array(shap_values).flatten()

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "SHAP Value": class_shap
        })
        importance_df["Absolute Impact"] = importance_df["SHAP Value"].abs()
        importance_df = importance_df.sort_values("Absolute Impact", ascending=False).head(10)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#E74C3C" if v < 0 else "#27AE60" for v in importance_df["SHAP Value"]]
        ax.barh(
            range(len(importance_df)),
            importance_df["SHAP Value"].values,
            color=colors,
            edgecolor="none",
            height=0.6
        )
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["Feature"].values, fontsize=10)
        ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
        ax.set_title(f"Top 10 features driving '{predicted_state}' prediction", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Plain language explanation
        top_feature = importance_df.iloc[0]
        direction = "increases" if top_feature["SHAP Value"] > 0 else "decreases"
        st.caption(
            f"The strongest driver is **{top_feature['Feature']}** which {direction} "
            f"the likelihood of being classified as **{predicted_state}**. "
            f"Green bars push toward this class, red bars push away from it."
        )

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
else:
    st.info("SHAP explainer not loaded — feature importance unavailable.")

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

    # Interpretation using BOTH risk_state AND lead_time
    if predicted_state == "Safe":
        st.success(f"**Stable:** This user is in a healthy engagement state. "
        f"No churn risk detected — lead-time prediction is not required. "
        f"Continue standard service delivery.")
    elif predicted_state == "At-Risk":
        if lead_time <= 7:
            st.error(f"**Critical:** Churn imminent — approximately {lead_time:.0f} days remaining")
        elif lead_time <= 30:
            st.error(f"⚠️ **Urgent:** Immediate intervention required — approximately {lead_time:.0f} days to churn")
        else:
            st.warning(f"**High Risk:** User is at-risk — approximately {lead_time:.0f} days to churn. Intervene now")

    elif predicted_state == "Warning":
        if lead_time <= 14:
            st.warning(f"⚠️ **Escalating:** Rapid decline detected — approximately {lead_time:.0f} days to churn")
        elif lead_time <= 30:
            st.warning(f"**Moderate Risk:** Plan intervention soon — approximately {lead_time:.0f} days to churn")
        else:
            st.warning(f"**Early Warning:** Gradual decline detected — approximately {lead_time:.0f} days to potential churn. Monitor closely")

else:
    st.info(
        "**Stable:** This user is in a healthy engagement state. "
        "No churn risk detected — lead-time prediction is not required. "
        "Continue standard service delivery."
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
