import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 
import joblib

# Safe import for XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Rainfall-Runoff Modeling Dashboard",
    page_icon="🌧️",
    layout="wide"
)

# -------------------------------------------------
# CONSTANTS & FEATURE LIST
# -------------------------------------------------
FEATURES = [
    "rainfall_mm", "temperature_c", "humidity_pct", "pressure_hpa",
    "wind_speed_kmh", "soil_moisture_pct", "evaporation_mm",
    "previous_flow_m3s", "rainfall_lag_1", "rainfall_lag_2",
    "rainfall_lag_3", "flow_lag_1", "flow_lag_2", "flow_lag_3"
]

TARGET = "runoff_m3s"

# -------------------------------------------------
# DATA & MODEL LOADING
# -------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("rainfall_runoff.csv"):
        df = pd.read_csv("rainfall_runoff.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        return df
    return None

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Linear Regression": ["linear_regression.pkl", "models/linear_regression.pkl"],
        "Random Forest": ["random_forest.pkl", "models/random_forest.pkl"],
        "XGBoost": ["xgboost.pkl", "models/xgboost.pkl"],
        "GPR": ["gpr.pkl", "models/gpr.pkl"]
    }

    for model_name, paths in model_files.items():
        if model_name == "XGBoost" and xgb is None:
            continue
        for path in paths:
            if os.path.exists(path):
                try:
                    models[model_name] = joblib.load(path)
                    break
                except Exception:
                    pass
    return models

df = load_data()
models = load_models()

# -------------------------------------------------
# SIDEBAR NAVIGATION & INPUTS
# -------------------------------------------------
st.sidebar.title("🌧️ Navigation")

# Unique key added to prevent DuplicateElementId error
page = st.sidebar.radio(
    "Go to", 
    ["Overview & Data", "Make Prediction", "Model Evaluation & Metrics"],
    key="unique_nav_radio"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Loaded Models")
if models:
    for m in models.keys():
        st.sidebar.success(f"✓ {m}")
else:
    st.sidebar.error("No trained models loaded.")

# -------------------------------------------------
# PAGE 1: OVERVIEW & DATA EDA
# -------------------------------------------------
if page == "Overview & Data":
    st.title("🌊 Rainfall-Runoff Modeling Dashboard")
    st.markdown("Analyze hydrologic data, explore feature correlations, and predict surface runoff.")

    if df is not None:
        st.subheader("📊 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Max Rainfall (mm)", f"{df['rainfall_mm'].max():.1f}" if 'rainfall_mm' in df.columns else "N/A")
        col3.metric("Max Runoff (m³/s)", f"{df['runoff_m3s'].max():.1f}" if 'runoff_m3s' in df.columns else "N/A")
        col4.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}" if 'date' in df.columns else "N/A")

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Time Series Data", "Rainfall vs Runoff", "Correlation Heatmap"])

        with tab1:
            st.subheader("Time Series Explorer")
            if "date" in df.columns:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=df["date"], y=df["runoff_m3s"], name="Runoff (m³/s)", line=dict(color="blue")))
                fig_ts.add_trace(go.Scatter(x=df["date"], y=df["rainfall_mm"], name="Rainfall (mm)", line=dict(color="cyan"), yaxis="y2"))
                
                fig_ts.update_layout(
                    title="Rainfall and Runoff Over Time",
                    yaxis=dict(title="Runoff (m³/s)"),
                    yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.write(df.head())

        with tab2:
            st.subheader("Rainfall vs Runoff Scatter Analysis")
            if "rainfall_mm" in df.columns and "runoff_m3s" in df.columns:
                fig_scatter = px.scatter(
                    df, x="rainfall_mm", y="runoff_m3s",
                    color="soil_moisture_pct" if "soil_moisture_pct" in df.columns else None,
                    title="Rainfall (mm) vs Runoff (m³/s)",
                    trendline="ols",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.subheader("Feature Correlation Matrix")
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

    else:
        st.warning("⚠️ `rainfall_runoff.csv` file not found in current directory.")

# -------------------------------------------------
# PAGE 2: MAKE PREDICTIONS
# -------------------------------------------------
elif page == "Make Prediction":
    st.title("🔮 Runoff Prediction Tool")
    st.markdown("Adjust environmental variables below to compute real-time runoff forecasts.")

    if not models:
        st.error("No loaded models found. Please train models first.")
    else:
        selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
        active_model = models[selected_model_name]

        st.markdown("---")
        st.subheader("Set Input Parameters")

        input_data = {}
        cols = st.columns(3)

        for idx, feat in enumerate(FEATURES):
            col = cols[idx % 3]
            min_val = float(df[feat].min()) if (df is not None and feat in df.columns) else 0.0
            max_val = float(df[feat].max()) if (df is not None and feat in df.columns) else 100.0
            mean_val = float(df[feat].mean()) if (df is not None and feat in df.columns) else 10.0

            input_data[feat] = col.number_input(
                label=f"{feat}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"input_{feat}"
            )

        input_df = pd.DataFrame([input_data])

        st.markdown("---")
        if st.button("🚀 Calculate Predicted Runoff", type="primary"):
            try:
                prediction = active_model.predict(input_df)[0]
                st.balloons()
                st.success(f"### Predicted Surface Runoff: **{prediction:.3f} m³/s**")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# -------------------------------------------------
# PAGE 3: MODEL EVALUATION & METRICS
# -------------------------------------------------
elif page == "Model Evaluation & Metrics":
    st.title("📈 Model Performance & Metrics")

    metrics_path = "model_comparison.csv"
    if os.path.exists(metrics_path):
        st.subheader("🏆 Model Comparison Table")
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = px.bar(metrics_df, x="Model", y="RMSE", color="Model", title="RMSE by Model (Lower is Better)")
            st.plotly_chart(fig_rmse, use_container_width=True)
        with col2:
            fig_r2 = px.bar(metrics_df, x="Model", y="R2", color="Model", title="R² Score by Model (Higher is Better)")
            st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    if "Random Forest" in models:
        st.subheader("🌲 Random Forest Feature Importance")
        rf_model = models["Random Forest"]

        if hasattr(rf_model, "named_steps") and "model" in rf_model.named_steps:
            rf_estimator = rf_model.named_steps["model"]
        else:
            rf_estimator = rf_model

        if hasattr(rf_estimator, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": rf_estimator.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig_imp = px.bar(
                importance_df, x="Importance", y="Feature", orientation="h",
                title="Random Forest Feature Importance", color="Importance",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
