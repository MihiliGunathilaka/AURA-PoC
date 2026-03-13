from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb


# FastAPI App


app = FastAPI(
    title="AURA Prediction API",
    description="Churn risk + lead-time prediction backend",
    version="2.1-FIXED"
)


# Request Schema


class UserSnapshot(BaseModel):
    region: str
    subscription_tier: str
    tenure_days: float

    login_count: float
    session_duration_minutes: float
    api_calls: float
    feature_usage_score: float
    resource_consumption_mb: float
    cpu_usage_percent: float
    support_tickets_created: float

    login_avg_7d: float
    session_avg_7d: float
    login_trend_7d: float
    engagement_score: float

    inactivity_days: float
    usage_drop_pct: float
    volatility_score: float
    resource_efficiency: float



# Load Model 1 (Risk Classifier)


print(" Loading Model 1 (Risk Classifier)...")

MODEL1_PATH = "AURA_xgboost_model.pkl"
FEATURES_PATH = "AURA_feature_names.txt"

model1 = None
FEATURE_NAMES = []

try:
    model1 = joblib.load(MODEL1_PATH)
    with open(FEATURES_PATH) as f:
        FEATURE_NAMES = [l.strip() for l in f if l.strip()]
    print(f" Model 1 loaded ({len(FEATURE_NAMES)} features)")
except Exception as e:
    print(f" Failed to load Model 1: {e}")

LABEL_MAP = {0: "Safe", 1: "Warning", 2: "At-Risk"}



# Load Model 2 (Lead-Time Model)


print(" Loading Model 2 (Lead Time Model)...")

MODEL2_PATH = "AURA_model2_final.pkl"
model2_package = None
model2_model = None
model2_scaler = None
model2_features = None
model2_bias = 0.0

try:
    if os.path.exists(MODEL2_PATH):
        model2_package = joblib.load(MODEL2_PATH)
        
        # Extract components
        model2_model = model2_package.get("model")
        model2_scaler = model2_package.get("scaler")
        model2_features = model2_package.get("features")
        model2_bias = model2_package.get("bias_correction", 0.0)
        
        print(f" Model 2 loaded")
        print(f"   Features: {len(model2_features) if model2_features else 0}")
        print(f"   Bias correction: {model2_bias:.2f} days")
    else:
        raise FileNotFoundError(MODEL2_PATH)

except Exception as e:
    print(f"⚠️ Model 2 not loaded: {e}")
    model2_package = None



# Feature Engineering for Model 2


def engineer_features_model2(snapshot: UserSnapshot):
    """
    Create the SAME engineered features as training.
    This is CRITICAL - must match exactly!
    """
    # Start with raw data as DataFrame
    data = {
        'region': snapshot.region,
        'subscription_tier': snapshot.subscription_tier,
        'tenure_days': snapshot.tenure_days,
        'login_count': snapshot.login_count,
        'session_duration_minutes': snapshot.session_duration_minutes,
        'api_calls': snapshot.api_calls,
        'feature_usage_score': snapshot.feature_usage_score,
        'resource_consumption_mb': snapshot.resource_consumption_mb,
        'cpu_usage_percent': snapshot.cpu_usage_percent,
        'support_tickets_created': snapshot.support_tickets_created,
        'login_avg_7d': snapshot.login_avg_7d,
        'session_avg_7d': snapshot.session_avg_7d,
        'login_trend_7d': snapshot.login_trend_7d,
        'engagement_score': snapshot.engagement_score,
        'inactivity_days': snapshot.inactivity_days,
        'usage_drop_pct': snapshot.usage_drop_pct,
        'volatility_score': snapshot.volatility_score,
        'resource_efficiency': snapshot.resource_efficiency
    }
    
    df = pd.DataFrame([data])
    
    
    # ENGINEERED FEATURES (Same as training!)
    
    
    # Core interactions
    if 'login_avg_7d' in df.columns and 'session_avg_7d' in df.columns:
        df['session_per_login'] = df['session_avg_7d'] / (df['login_avg_7d'] + 0.1)
        df['activity_score'] = df['login_avg_7d'] + df['session_avg_7d']
        df['login_session_interaction'] = df['login_avg_7d'] * df['session_avg_7d']
    
    if 'engagement_score' in df.columns and 'session_avg_7d' in df.columns:
        df['engagement_efficiency'] = df['engagement_score'] / (df['session_avg_7d'] + 0.1)
    
    if 'feature_usage_score' in df.columns and 'login_avg_7d' in df.columns:
        df['feature_adoption'] = df['feature_usage_score'] / (df['login_avg_7d'] + 0.1)
    
    # Polynomial features (squares)
    for feat in ['login_avg_7d', 'session_avg_7d']:
        if feat in df.columns:
            df[f"{feat}_squared"] = df[feat] ** 2
    
    # Velocity aggregates (if you have velocity columns)
    # Note: Your input schema doesn't include velocity columns
    # If they exist in the full dataset, add them here
    
    # Volatility aggregates (if you have volatility columns)
    # Same note as velocity
    
    # Clean up inf/nan
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


# Feature Builder for Model 1


def build_feature_vector_model1(snapshot: UserSnapshot, feature_names):
    """Build feature vector for Model 1 (Risk Classifier)"""
    data = snapshot.dict()
    row = []

    for feat in feature_names:
        if feat.startswith("region_"):
            row.append(1.0 if data["region"] == feat.replace("region_", "") else 0.0)
        elif feat.startswith("subscription_tier_"):
            row.append(1.0 if data["subscription_tier"] == feat.replace("subscription_tier_", "") else 0.0)
        else:
            row.append(float(data.get(feat, 0.0)))

    return pd.DataFrame([row], columns=feature_names)



# Root Endpoint


@app.get("/")
def root():
    return {
        "status": "AURA API running (FIXED for Model 2)",
        "model_1_loaded": model1 is not None,
        "model_2_loaded": model2_model is not None,
        "version": "2.1-FIXED"
    }


# Prediction Endpoint


@app.post("/predict")
def predict(snapshot: UserSnapshot):

    if model1 is None:
        return {"error": "Model 1 not available"}

    
    # STEP 1: Risk Prediction (Model 1)
    
    
    X1 = build_feature_vector_model1(snapshot, FEATURE_NAMES)
    probs = model1.predict_proba(X1)[0]
    label = int(np.argmax(probs))

    risk_state = LABEL_MAP[label]

    response = {
        "risk_state": risk_state,
        "risk_probabilities": {
            "Safe": float(probs[0]),
            "Warning": float(probs[1]),
            "At-Risk": float(probs[2]),
        },
        "lead_time_days": None,
        "confidence_interval_lower": None,
        "confidence_interval_upper": None
    }

    
    # STEP 2: Lead Time Prediction (Model 2)
    
    
    if model2_model is not None and model2_scaler is not None and model2_features is not None:
        try:
            # Engineer features (CRITICAL!)
            df_engineered = engineer_features_model2(snapshot)
            
            # Select only the features used in training
            # Handle missing features gracefully
            X2 = pd.DataFrame()
            for feat in model2_features:
                if feat in df_engineered.columns:
                    X2[feat] = df_engineered[feat]
                else:
                    # Feature doesn't exist, fill with 0
                    X2[feat] = 0.0
            
            # Scale features
            X2_scaled = model2_scaler.transform(X2)
            
            # Build DMatrix for XGBoost AFT
            dmat = xgb.DMatrix(X2_scaled, feature_names=model2_features)
            
            # Predict
            pred_raw = model2_model.predict(dmat)[0]
            
            # Apply bias correction and clip
            lead_time = pred_raw + model2_bias
            lead_time = np.clip(lead_time, 1, 120)
            
            # Calculate confidence interval (±2 MAE = ±42 days for 95% CI)
            mae = 20.85  # From your test results
            ci_lower = max(1, lead_time - 2 * mae)
            ci_upper = min(120, lead_time + 2 * mae)
            
            response["lead_time_days"] = round(float(lead_time), 2)
            response["confidence_interval_lower"] = round(float(ci_lower), 2)
            response["confidence_interval_upper"] = round(float(ci_upper), 2)
            
            print(f" Lead time prediction: {lead_time:.2f} days")

        except Exception as e:
            print(f" Lead time prediction failed: {e}")
            import traceback
            traceback.print_exc()

    return response



# Health Check Endpoint


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {
            "risk_classifier": "loaded" if model1 is not None else "not loaded",
            "lead_time_predictor": "loaded" if model2_model is not None else "not loaded"
        },
        "model_2_info": {
            "features_count": len(model2_features) if model2_features else 0,
            "bias_correction": float(model2_bias),
            "scaler_loaded": model2_scaler is not None
        }
    }



# Debug Endpoint (Optional - for troubleshooting)


@app.post("/debug/features")
def debug_features(snapshot: UserSnapshot):
    """
    Debug endpoint to see engineered features
    """
    if model2_features is None:
        return {"error": "Model 2 not loaded"}
    
    # Engineer features
    df_eng = engineer_features_model2(snapshot)
    
    # Show which features exist
    available_features = {
        feat: feat in df_eng.columns 
        for feat in model2_features
    }
    
    # Get sample values for first 10 features
    sample_values = {}
    for feat in model2_features[:10]:
        if feat in df_eng.columns:
            sample_values[feat] = float(df_eng[feat].iloc[0])
        else:
            sample_values[feat] = None
    
    return {
        "total_features_needed": len(model2_features),
        "features_available": sum(available_features.values()),
        "features_missing": len(model2_features) - sum(available_features.values()),
        "sample_values": sample_values,
        "available_features": available_features
    }