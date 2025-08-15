
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Heart Disease Prediction UI", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction")

# ====== SCHEMA (must match training) ======
REQUIRED_INPUT = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal"
]
NUMERIC_COLS = ["age","trestbps","chol","thalach","oldpeak","sex","fbs","exang"]
CAT_COLS     = ["cp","restecg","slope","ca","thal"]
RENAME_MAP = {}

# Human-readable labels
LABELS = {
    "sex":   {0: "Female", 1: "Male"},
    "cp":    {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy (LVH)"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal":  {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
    "fbs":   {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
}
INV = {k:{v:k for k,v in d.items()} for k,d in LABELS.items()}

# ====== Custom cleaner (must exist for pickle) ======
class SchemaCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_ = X.copy()
        if hasattr(X_, "replace"):
            X_.replace("?", np.nan, inplace=True)
        X_.columns = X_.columns.str.strip()
        if RENAME_MAP:
            X_.rename(columns=RENAME_MAP, inplace=True)
        for c in REQUIRED_INPUT:
            if c not in X_.columns:
                X_[c] = np.nan
        X_ = X_[REQUIRED_INPUT]
        for c in NUMERIC_COLS:
            X_[c] = pd.to_numeric(X_[c], errors="coerce")
        for c in CAT_COLS:
            if X_[c].isna().all():
                X_[c] = "missing"
        return X_

def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if RENAME_MAP:
        df.rename(columns=RENAME_MAP, inplace=True)
    for c in REQUIRED_INPUT:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_INPUT]

# ====== Load model (no auto predict) ======
@st.cache_resource(show_spinner=False)
def load_model(path: str = "heart_pipeline.pkl"):
    if not os.path.exists(path):
        return None, f"Model file '{path}' was not found."
    try:
        mdl = joblib.load(path)  # resolves SchemaCleaner
        return mdl, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

model, model_err = load_model()
if model_err:
    st.warning(model_err)

# ====== Optional CSV for schema/auto-fill only (no auto prediction) ======
st.sidebar.header("📁 Optional CSV")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
df = None
target_col = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded data with shape {df.shape}")
        cols = list(df.columns)
        target_col = st.sidebar.selectbox("Target column (ignored for prediction)", options=["(none)"] + cols, index=0)
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")

# ====== Input mode (no auto compute) ======
st.subheader("⚙️ Enter Information & Press **Get Diagnosis**")

input_mode = st.radio("Input mode", ["Manual input", "Pick a row from uploaded data"], horizontal=True)

# Prefill helpers
def pick_default(col, fallback):
    if df is not None and col in (df.columns if target_col is None else [c for c in df.columns if c != target_col]) and pd.api.types.is_numeric_dtype(df[col]):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            return float(vals.median())
    return fallback

# ====== FORM: user edits then presses a button ======
with st.form("predict_form", clear_on_submit=False):
    if input_mode == "Pick a row from uploaded data":
        if df is None:
            st.warning("Upload a CSV first to use this mode.")
            st.stop()
        idx = st.number_input("Row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
        row = df.iloc[[idx]].copy()
        if target_col and target_col in row.columns and target_col != "(none)":
            row = row.drop(columns=[target_col])

        # Extract defaults from row (fallbacks if missing)
        def get_row_val(col, fallback=None):
            return row.iloc[0].get(col, fallback) if col in row.columns else fallback

        # NUMERIC
        age      = st.number_input("Age (years)", value=float(get_row_val("age", pick_default("age", 50.0)) or 50.0))
        trestbps = st.number_input("Resting blood pressure (mm Hg)", value=float(get_row_val("trestbps", pick_default("trestbps", 130.0)) or 130.0))
        chol     = st.number_input("Serum cholesterol (mg/dl)", value=float(get_row_val("chol", pick_default("chol", 240.0)) or 240.0))
        thalach  = st.number_input("Max heart rate achieved", value=float(get_row_val("thalach", pick_default("thalach", 150.0)) or 150.0))
        oldpeak  = st.number_input("ST depression (oldpeak)", value=float(get_row_val("oldpeak", pick_default("oldpeak", 1.0)) or 1.0))
        ca       = st.number_input("Major vessels colored by fluoroscopy (0–3)", min_value=0, max_value=3, value=int(get_row_val("ca", 0) or 0), step=1)

        # CATEGORICAL (map row codes -> label)
        def idx_for(col, mapping, default_code, row_code):
            code = int(row_code) if pd.notna(row_code) else default_code
            label = mapping.get(code, list(mapping.values())[0])
            return list(mapping.values()).index(label)

        sex_label   = st.selectbox("Sex", options=list(LABELS["sex"].values()),
                                   index=idx_for("sex", LABELS["sex"], 1, get_row_val("sex", 1)))
        cp_label    = st.selectbox("Chest pain type", options=list(LABELS["cp"].values()),
                                   index=idx_for("cp", LABELS["cp"], 4, get_row_val("cp", 4)))
        rest_label  = st.selectbox("Resting ECG", options=list(LABELS["restecg"].values()),
                                   index=idx_for("restecg", LABELS["restecg"], 0, get_row_val("restecg", 0)))
        exang_label = st.selectbox("Exercise-induced angina", options=list(LABELS["exang"].values()),
                                   index=idx_for("exang", LABELS["exang"], 0, get_row_val("exang", 0)))
        slope_label = st.selectbox("ST slope at peak exercise", options=list(LABELS["slope"].values()),
                                   index=idx_for("slope", LABELS["slope"], 2, get_row_val("slope", 2)))
        thal_label  = st.selectbox("Thalassemia test", options=list(LABELS["thal"].values()),
                                   index=idx_for("thal", LABELS["thal"], 3, get_row_val("thal", 3)))
        fbs_label   = st.selectbox("Fasting blood sugar", options=list(LABELS["fbs"].values()),
                                   index=idx_for("fbs", LABELS["fbs"], 0, get_row_val("fbs", 0)))

    else:
        # MANUAL
        age      = st.number_input("Age (years)", value=int(pick_default("age", 50.0)), step=1)
        trestbps = st.number_input("Resting blood pressure (mm Hg)", value=int(pick_default("trestbps", 130.0)), step=1)
        chol     = st.number_input("Serum cholesterol (mg/dl)", value=int(pick_default("chol", 240.0)), step=1)
        thalach  = st.number_input("Max heart rate achieved", value=int(pick_default("thalach", 150.0)), step=1)
        oldpeak  = st.number_input("ST depression (oldpeak)", value=float(pick_default("oldpeak", 1.0)))
        ca       = st.number_input("Major vessels colored by fluoroscopy (0–3)", min_value=0, max_value=3, value=0, step=1)

        sex_label   = st.selectbox("Sex", options=list(LABELS["sex"].values()), index=1)
        cp_label    = st.selectbox("Chest pain type", options=list(LABELS["cp"].values()))
        rest_label  = st.selectbox("Resting ECG", options=list(LABELS["restecg"].values()))
        exang_label = st.selectbox("Exercise-induced angina", options=list(LABELS["exang"].values()), index=0)
        slope_label = st.selectbox("ST slope at peak exercise", options=list(LABELS["slope"].values()))
        thal_label  = st.selectbox("Thalassemia test", options=list(LABELS["thal"].values()))
        fbs_label   = st.selectbox("Fasting blood sugar", options=list(LABELS["fbs"].values()), index=0)

    submitted = st.form_submit_button("🩺 Get diagnosis")

# ====== Predict ONLY when button is pressed ======
if submitted:
    if model is None:
        st.error("Model not loaded.")
    else:
        # Convert labels back to numeric codes
        values = {
            "age": float(age),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "thalach": float(thalach),
            "oldpeak": float(oldpeak),
            "ca": int(ca),
            "sex": INV["sex"][sex_label],
            "cp": INV["cp"][cp_label],
            "restecg": INV["restecg"][rest_label],
            "exang": INV["exang"][exang_label],
            "slope": INV["slope"][slope_label],
            "thal": INV["thal"][thal_label],
            "fbs": INV["fbs"][fbs_label],
        }
        X = pd.DataFrame([[values.get(c) for c in REQUIRED_INPUT]], columns=REQUIRED_INPUT)
        X = prepare_for_model(X)

        try:
            y_pred = model.predict(X)[0]
            pred_text = "No disease" if str(y_pred) == "0" else "Disease"
            st.success(f"Prediction: **{pred_text}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Tip: To try again, just edit any field and press the button again — no auto prediction.
