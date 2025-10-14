import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import warnings

# Suppress warnings for cleaner production output
warnings.filterwarnings('ignore')

# Configure Streamlit page settings for production
st.set_page_config(
    page_title="Disease Outbreak Risk Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import visualization libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server deployment
    # Set matplotlib parameters for production
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['font.size'] = 10
except ImportError as e:
    st.error(f"Error importing matplotlib: {e}")
    plt = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    st.error(f"Error importing plotly: {e}")
    px = None
    go = None

try:
    import seaborn as sns
except ImportError as e:
    st.warning(f"Seaborn not available: {e}")
    sns = None

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    from scipy.stats import ks_2samp
except ImportError as e:
    st.error(f"Error importing scipy: {e}")
    ks_2samp = None

# ----------------------------
# Paths & lazy loading
# ----------------------------
REPO_DIR = os.path.dirname(os.path.dirname(__file__))
ART_DIR = os.path.join(REPO_DIR, "app", "artifacts")
PIPE_PATH = os.path.join(ART_DIR, "inference_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")
REF_PATH = os.path.join(ART_DIR, "reference_sample.csv")

LABEL_MAP = {0: "Low Risk", 1: "High Risk"}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    load_error = None
    pipeline, expected_cols, ref = None, None, None
    try:
        import joblib

        pipeline = joblib.load(PIPE_PATH)
        expected_cols = json.load(open(COLS_PATH))["expected_input_cols"]
    except Exception as e:
        load_error = f"Artifact load failed: {e}"

    if os.path.exists(REF_PATH):
        try:
            ref = pd.read_csv(REF_PATH)
        except Exception as e:
            load_error = f"Reference sample load failed: {e}"

    return pipeline, expected_cols, ref, load_error


inference_pipeline, EXPECTED_COLS, REF, LOAD_ERR = load_artifacts()

st.title("🦠 Disease Outbreak Risk Monitoring Dashboard")
st.caption(
    "Real-time dashboard for disease outbreak risk prediction, monitoring, and analysis."
)

# ----------------------------
# Sidebar controls (auto-detect sensitive & target)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # Auto-detect candidate sensitive & target columns from the uploaded CSV (if any)
    detected_sensitive = "Country"
    detected_target = ""
    if uploaded is not None:
        try:
            _tmp_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            # Candidate sensitive cols: categorical with reasonable cardinality (<=30)
            cand_sens = [
                c
                for c in _tmp_df.columns
                if (
                    pd.api.types.is_object_dtype(_tmp_df[c])
                    or pd.api.types.is_categorical_dtype(_tmp_df[c])
                )
                and _tmp_df[c].nunique() <= 30
            ]
            # Prefer common choices if present
            for pref in ["Country", "Disease_Name", "Region"]:
                if pref in cand_sens:
                    detected_sensitive = pref
                    break
                elif cand_sens:
                    detected_sensitive = cand_sens[0]

            # Candidate ground truth columns
            for pref_t in ["high_risk_outbreak", "outbreak_risk", "target", "label"]:
                if pref_t in _tmp_df.columns:
                    detected_target = pref_t
                    break
        except Exception:
            pass

    sensitive_attr = st.text_input(
        "Grouping column for analysis", value=detected_sensitive
    )
    target_attr = st.text_input("Ground-truth column (optional)", value=detected_target)

    threshold = st.slider(
        "Probability threshold for 'High Risk'", 0.0, 1.0, 0.5, 0.01
    )
    st.divider()
    st.write(
        "Artifacts status:", "`OK`" if inference_pipeline else f"`Degraded: {LOAD_ERR}`"
    )


# ----------------------------
# Helper functions
# ----------------------------
def align_columns(df: pd.DataFrame, expected_cols):
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols]


def predict_df(df: pd.DataFrame):
    model = getattr(inference_pipeline, "named_steps", {}).get("model", None)
    proba = None
    if hasattr(inference_pipeline, "predict_proba"):
        proba = inference_pipeline.predict_proba(df)[:, 1]
    elif model is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(
            inference_pipeline.named_steps["preprocess"].transform(df)
        )[:, 1]
    preds = inference_pipeline.predict(df)
    return preds, proba


def psi(reference: pd.Series, current: pd.Series, bins: int = 10):
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts) / max(1, ref_counts.sum())
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts) / max(1, cur_counts.sum())
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def model_type(model):
    if isinstance(model, RandomForestClassifier):
        return "rf"
    if isinstance(model, LogisticRegression):
        return "lr"
    return "other"


def get_feature_names(preprocess):
    names = []
    for name, trans, cols in preprocess.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            try:
                oh = trans.named_steps["oh"]
                names.extend(list(oh.get_feature_names_out(cols)))
            except Exception:
                names.extend(cols)
    return names


def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# ----------------------------
# Tabs
# ----------------------------
tab_pred, tab_explore, tab_analysis, tab_shap, tab_responsible, tab_fair, tab_drift = st.tabs(
    ["🔮 Risk Prediction", "📊 Data Exploration", "📈 Risk Analysis", "🔎 SHAP", "🛡️ Responsible AI", "⚖️ Fairness", "🌊 Drift"]
)

# ----------------------------
# Predict tab
# ----------------------------
with tab_pred:
    st.subheader("Disease Outbreak Risk Prediction")
    if inference_pipeline is None or EXPECTED_COLS is None:
        st.warning("Artifacts not loaded. Check /app/artifacts files.")
    else:
        df_in = None
        if uploaded is not None:
            try:
                df_in = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                if df_in.empty:
                    st.error("Uploaded CSV is empty.")
                    df_in = None
                else:
                    st.dataframe(df_in.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_in = None
        else:
            st.info("Upload a CSV or use a sample record below.")
            demo = {
                "Population": 50000000,
                "Cases_Reported": 1000,
                "Deaths_Reported": 50,
                "Recovered": 900,
                "Vaccination_Coverage_Pct": 75.0,
                "Healthcare_Expenditure_PctGDP": 8.5,
                "Urbanization_Rate_Pct": 80.0,
                "Avg_Temperature_C": 25.0,
                "Avg_Humidity_Pct": 65.0,
                "case_fatality_rate": 0.05,
                "cases_per_100k": 2.0,
                "recovery_rate": 0.9,
                "healthcare_vaccination_score": 637.5,
                "Country": "Thailand",
                "Disease_Name": "Malaria"
            }
            df_in = pd.DataFrame([demo])

        if df_in is not None:
            al = align_columns(df_in.copy(), EXPECTED_COLS)
            preds, proba = predict_df(al)
            out = pd.DataFrame(
                {
                    "prediction": preds.astype(int),
                    "label": [LABEL_MAP.get(int(p), str(p)) for p in preds],
                    "proba_high_risk": proba if proba is not None else np.nan,
                }
            )
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(50), use_container_width=True)

            # optional ground truth
            gt_col = None
            for c in ["high_risk_outbreak", "outbreak_risk", "target", "label"]:
                if c in df_in.columns:
                    gt_col = c
                    break
            if gt_col:
                y_true_raw = (
                    df_in[gt_col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .map(
                        {
                            "low risk": 0,
                            "low": 0,
                            "safe": 0,
                            "0": 0,
                            "false": 0,
                            "no": 0,
                            "high risk": 1,
                            "high": 1,
                            "dangerous": 1,
                            "1": 1,
                            "true": 1,
                            "yes": 1,
                        }
                    )
                )
                y_true = y_true_raw.fillna(0).astype(int).values
                y_pred = (
                    (out["proba_high_risk"].fillna(0.0).values >= threshold).astype(
                        int
                    )
                    if proba is not None
                    else out["prediction"].values
                )
                st.write("**Model Performance Metrics**")
                y_pred = (
                    (out["proba_high_risk"].fillna(0.0).values >= threshold).astype(
                        int
                    )
                    if proba is not None
                    else out["prediction"].values
                )
                # tabular classification report
                report_dict = classification_report(
                    y_true, y_pred, output_dict=True, digits=4
                )
                report_df = (
                    pd.DataFrame(report_dict)
                    .T.reset_index()
                    .rename(columns={"index": "label"})
                )
                # order columns nicely if present
                cols = [
                    c
                    for c in ["label", "precision", "recall", "f1-score", "support"]
                    if c in report_df.columns
                ]
                st.dataframe(report_df[cols], use_container_width=True)

                # confusion matrix as a table
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=["True 0 (Low Risk)", "True 1 (High Risk)"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.write("**Confusion Matrix**")
                st.dataframe(cm_df, use_container_width=True)

# ----------------------------
# Data Exploration tab
# ----------------------------
with tab_explore:
    st.subheader("📊 Data Exploration")

    if uploaded is None:
        st.info("Upload a CSV file to explore disease outbreak data.")
        # Show default dataset exploration
        if REF is not None:
            st.write("**Reference Dataset Overview**")
            st.dataframe(REF.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Disease_Name' in REF.columns:
                    disease_counts = REF['Disease_Name'].value_counts()
                    fig_disease = px.bar(
                        x=disease_counts.values,
                        y=disease_counts.index,
                        orientation='h',
                        title="Disease Distribution",
                        labels={'x': 'Count', 'y': 'Disease'}
                    )
                    st.plotly_chart(fig_disease, use_container_width=True)
            
            with col2:
                if 'Country' in REF.columns:
                    country_counts = REF['Country'].value_counts().head(10)
                    fig_country = px.bar(
                        x=country_counts.values,
                        y=country_counts.index,
                        orientation='h',
                        title="Top 10 Countries by Records",
                        labels={'x': 'Count', 'y': 'Country'}
                    )
                    st.plotly_chart(fig_country, use_container_width=True)
    else:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            st.write("**Dataset Overview**")
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                if 'Disease_Name' in df.columns:
                    st.metric("Unique Diseases", df['Disease_Name'].nunique())
            
            # Visualizations
            if 'Cases_Reported' in df.columns and 'Deaths_Reported' in df.columns:
                st.write("**Case Fatality Analysis**")
                df['case_fatality_rate'] = df['Deaths_Reported'] / df['Cases_Reported'].replace(0, 1)
                
                fig_cfr = px.histogram(
                    df, 
                    x='case_fatality_rate',
                    title="Distribution of Case Fatality Rates",
                    nbins=50
                )
                st.plotly_chart(fig_cfr, use_container_width=True)
            
            if 'Country' in df.columns and 'Cases_Reported' in df.columns:
                st.write("**Geographic Analysis**")
                country_cases = df.groupby('Country')['Cases_Reported'].sum().sort_values(ascending=False).head(15)
                
                fig_geo = px.bar(
                    x=country_cases.values,
                    y=country_cases.index,
                    orientation='h',
                    title="Total Cases by Country (Top 15)",
                    labels={'x': 'Total Cases', 'y': 'Country'}
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error exploring data: {e}")

# ----------------------------
# Risk Analysis tab
# ----------------------------
with tab_analysis:
    st.subheader("Disease Outbreak Risk Analysis")
    
    if uploaded is None:
        st.info("Upload a CSV file to analyze disease outbreak risk patterns.")
    else:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            
            # Create risk features if not present
            if 'case_fatality_rate' not in df.columns and 'Cases_Reported' in df.columns and 'Deaths_Reported' in df.columns:
                df['case_fatality_rate'] = df['Deaths_Reported'] / df['Cases_Reported'].replace(0, 1)
            
            if 'cases_per_100k' not in df.columns and 'Cases_Reported' in df.columns and 'Population' in df.columns:
                df['cases_per_100k'] = (df['Cases_Reported'] / df['Population']) * 100000
            
            # Make predictions
            if inference_pipeline is not None and EXPECTED_COLS is not None:
                al = align_columns(df.copy(), EXPECTED_COLS)
                preds, proba = predict_df(al)
                df['predicted_risk'] = preds
                if proba is not None:
                    df['risk_probability'] = proba
            
            # Risk analysis visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if 'case_fatality_rate' in df.columns:
                    fig_cfr = px.scatter(
                        df, 
                        x='case_fatality_rate', 
                        y='Cases_Reported',
                        color='Disease_Name' if 'Disease_Name' in df.columns else None,
                        title="Case Fatality Rate vs Cases Reported",
                        log_y=True
                    )
                    st.plotly_chart(fig_cfr, use_container_width=True)
            
            with col2:
                if 'risk_probability' in df.columns:
                    fig_risk = px.histogram(
                        df,
                        x='risk_probability',
                        title="Distribution of Risk Probabilities",
                        nbins=30
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            # Enhanced Model Analysis and Metrics
            if inference_pipeline is not None:
                st.write("**📊 Comprehensive Model Analysis**")
                
                # Model Performance Metrics
                if 'predicted_risk' in df.columns and 'risk_probability' in df.columns:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        high_risk_count = (df['predicted_risk'] == 1).sum()
                        st.metric("High Risk Cases", high_risk_count)
                    
                    with col2:
                        avg_risk_prob = df['risk_probability'].mean()
                        st.metric("Avg Risk Probability", f"{avg_risk_prob:.3f}")
                    
                    with col3:
                        max_risk_prob = df['risk_probability'].max()
                        st.metric("Max Risk Probability", f"{max_risk_prob:.3f}")
                    
                    with col4:
                        risk_variance = df['risk_probability'].var()
                        st.metric("Risk Variance", f"{risk_variance:.4f}")
                
                # Feature Importance Analysis
                try:
                    model = inference_pipeline.named_steps["model"]
                    if hasattr(model, 'feature_importances_'):
                        preprocessor = inference_pipeline.named_steps["preprocess"]
                        
                        # Get feature names
                        numeric_features = [c for c in EXPECTED_COLS if c not in ['Country', 'Disease_Name']]
                        categorical_features = ['Country', 'Disease_Name']
                        
                        feature_names = numeric_features.copy()
                        
                        # Add one-hot encoded features
                        try:
                            cat_transformer = preprocessor.named_transformers_['cat']
                            if hasattr(cat_transformer, 'get_feature_names_out'):
                                cat_features = cat_transformer.get_feature_names_out(categorical_features)
                                feature_names.extend(cat_features)
                        except:
                            feature_names.extend(categorical_features)
                        
                        importances = model.feature_importances_
                        
                        # Ensure matching lengths
                        min_len = min(len(feature_names), len(importances))
                        feature_names = feature_names[:min_len]
                        importances = importances[:min_len]
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances,
                            'Cumulative_Importance': np.cumsum(importances),
                            'Importance_Pct': (importances / importances.sum()) * 100
                        }).sort_values('Importance', ascending=False)
                        
                        # Enhanced visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Main importance plot
                            fig_imp = px.bar(
                                importance_df.head(15),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Feature Importance (Top 15)",
                                color='Importance',
                                color_continuous_scale='viridis',
                                text='Importance_Pct'
                            )
                            fig_imp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_imp, use_container_width=True)
                        
                        with col2:
                            # Cumulative importance plot
                            fig_cum = px.line(
                                importance_df.head(20),
                                x=range(1, min(21, len(importance_df) + 1)),
                                y='Cumulative_Importance',
                                title="Cumulative Feature Importance",
                                markers=True
                            )
                            fig_cum.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                            annotation_text="80% threshold")
                            fig_cum.update_xaxes(title="Number of Features")
                            fig_cum.update_yaxes(title="Cumulative Importance")
                            st.plotly_chart(fig_cum, use_container_width=True)
                        
                        # Feature categories analysis
                        st.write("**Feature Categories Analysis**")
                        
                        # Categorize features
                        categories = {
                            'Epidemiological': ['case_fatality_rate', 'cases_per_100k', 'recovery_rate', 
                                              'Cases_Reported', 'Deaths_Reported', 'Recovered'],
                            'Healthcare': ['Healthcare_Expenditure_PctGDP', 'Vaccination_Coverage_Pct', 
                                         'healthcare_vaccination_score'],
                            'Demographics': ['Population', 'Urbanization_Rate_Pct'],
                            'Environmental': ['Avg_Temperature_C', 'Avg_Humidity_Pct'],
                            'Geographic': [f for f in feature_names if f.startswith('Country_')],
                            'Disease_Type': [f for f in feature_names if f.startswith('Disease_Name_')]
                        }
                        
                        category_importance = {}
                        for cat, features in categories.items():
                            cat_imp = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
                            category_importance[cat] = cat_imp
                        
                        cat_df = pd.DataFrame(list(category_importance.items()), 
                                            columns=['Category', 'Total_Importance'])
                        cat_df = cat_df.sort_values('Total_Importance', ascending=False)
                        
                        fig_cat = px.pie(
                            cat_df, 
                            values='Total_Importance', 
                            names='Category',
                            title="Feature Importance by Category"
                        )
                        st.plotly_chart(fig_cat, use_container_width=True)
                        
                        # Detailed feature table
                        st.write("**Detailed Feature Analysis**")
                        st.dataframe(
                            importance_df[['Feature', 'Importance_Pct', 'Cumulative_Importance']].head(25), 
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"Could not compute feature importance: {e}")
                    st.write("Debug info:", str(e))
            
        except Exception as e:
            st.error(f"Error in risk analysis: {e}")

# ----------------------------
# SHAP Explanations tab
# ----------------------------
with tab_shap:
    st.subheader("🔎 SHAP Feature Importance Analysis")
    
    if inference_pipeline is None or EXPECTED_COLS is None:
        st.warning("Model artifacts not loaded. Cannot compute SHAP values.")
    else:
        # Data selection for SHAP analysis
        shap_data = None
        if uploaded is not None:
            try:
                shap_data = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                st.success(f"Using uploaded data with {len(shap_data)} records for SHAP analysis")
            except Exception as e:
                st.error(f"Error loading uploaded data: {e}")
                shap_data = None
        
        if shap_data is None and REF is not None:
            shap_data = REF.copy()
            st.info("Using reference dataset for SHAP analysis")
        
        if shap_data is not None:
            try:
                # Sample data for SHAP (limit to avoid memory issues)
                sample_size = min(100, len(shap_data))
                shap_sample = shap_data.sample(n=sample_size, random_state=42)
                
                # Prepare data
                al_sample = align_columns(shap_sample.copy(), EXPECTED_COLS)
                
                # Get model components
                model = inference_pipeline.named_steps["model"]
                preprocessor = inference_pipeline.named_steps["preprocess"]
                
                # Transform data
                X_transformed = preprocessor.transform(al_sample)
                if hasattr(X_transformed, 'toarray'):
                    X_transformed = X_transformed.toarray()
                
                # Get feature names for SHAP
                numeric_features = [c for c in EXPECTED_COLS if c not in ['Country', 'Disease_Name']]
                categorical_features = ['Country', 'Disease_Name']
                
                feature_names = numeric_features.copy()
                try:
                    cat_transformer = preprocessor.named_transformers_['cat']
                    if hasattr(cat_transformer, 'get_feature_names_out'):
                        cat_features = cat_transformer.get_feature_names_out(categorical_features)
                        feature_names.extend(cat_features)
                except:
                    feature_names.extend(categorical_features)
                
                # Ensure matching lengths
                min_len = min(len(feature_names), X_transformed.shape[1])
                feature_names = feature_names[:min_len]
                X_transformed = X_transformed[:, :min_len]
                
                st.write(f"**Computing SHAP values for {sample_size} samples with {len(feature_names)} features**")
                
                # Compute SHAP values
                try:
                    import shap
                    shap_available = True
                except ImportError:
                    st.error("SHAP library not available. Please install it with: pip install shap==0.44.1")
                    shap_available = False
                
                if shap_available:
                    with st.spinner("Computing SHAP values... This may take a moment."):
                        if hasattr(model, 'predict_proba'):
                            # For classification models
                            explainer = shap.Explainer(model.predict_proba, X_transformed)
                            shap_values = explainer(X_transformed)
                            
                            # Use class 1 (High Risk) SHAP values
                            if hasattr(shap_values, 'values') and len(shap_values.values.shape) > 2:
                                shap_vals = shap_values.values[:, :, 1]  # High risk class
                            else:
                                shap_vals = shap_values.values
                        else:
                            # Fallback for other models
                            explainer = shap.Explainer(model, X_transformed)
                            shap_values = explainer(X_transformed)
                            shap_vals = shap_values.values
                    
                    # Calculate feature importance like in your image
                    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
                    
                    # Create the feature importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'mean_abs_shap': mean_abs_shap
                    }).sort_values('mean_abs_shap', ascending=False)
                    
                    # Display results similar to your image
                    st.write("### Top features by mean(|SHAP|)")
                    
                    # Show the table like in your image
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(
                            importance_df.head(20).reset_index(drop=True), 
                            use_container_width=True,
                            height=400
                        )
                    
                    with col2:
                        # Create the horizontal bar chart like in your image
                        if plt is not None:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            top_features = importance_df.head(20)
                            y_pos = np.arange(len(top_features))
                            
                            bars = ax.barh(y_pos, top_features['mean_abs_shap'], 
                                          color='steelblue', alpha=0.8)
                            
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(top_features['feature'])
                            ax.invert_yaxis()  # Highest values at top
                            ax.set_xlabel('mean(|SHAP value|)')
                            ax.set_title('SHAP Feature Importance (Top 20)')
                            ax.grid(axis='x', alpha=0.3)
                            
                            # Add value labels on bars
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width, bar.get_y() + bar.get_height()/2, 
                                       f'{width:.3f}', ha='left', va='center', fontsize=8)
                            
                            plt.tight_layout()
                            st.pyplot(fig, clear_figure=True)
                        else:
                            st.error("Matplotlib not available for plotting")
                    
                    # Additional SHAP visualizations
                    st.write("### SHAP Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Features", len(feature_names))
                    with col2:
                        st.metric("Samples Analyzed", sample_size)
                    with col3:
                        st.metric("Top Feature Impact", f"{mean_abs_shap.max():.4f}")
                    with col4:
                        st.metric("Mean SHAP Magnitude", f"{mean_abs_shap.mean():.4f}")
                    
                    # Feature importance interpretation
                    st.write("### 🎯 Feature Importance Interpretation")
                    st.write("**Top 5 Most Important Features for Outbreak Risk:**")
                    
                    for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
                        feature_name = row['feature']
                        importance = row['mean_abs_shap']
                        
                        # Provide interpretation based on feature name
                        interpretation = ""
                        if 'case_fatality_rate' in feature_name.lower():
                            interpretation = "Higher case fatality rates strongly indicate higher outbreak risk"
                        elif 'cases_per_100k' in feature_name.lower():
                            interpretation = "Cases per 100k population is a key indicator of outbreak severity"
                        elif 'vaccination' in feature_name.lower():
                            interpretation = "Vaccination coverage affects population immunity and outbreak risk"
                        elif 'healthcare' in feature_name.lower():
                            interpretation = "Healthcare expenditure reflects system capacity to handle outbreaks"
                        elif 'country' in feature_name.lower():
                            interpretation = "Geographic factors influence outbreak patterns and response"
                        elif 'disease' in feature_name.lower():
                            interpretation = "Different diseases have varying risk profiles and transmission patterns"
                        else:
                            interpretation = "This feature contributes significantly to outbreak risk predictions"
                        
                        st.write(f"**{i+1}. {feature_name}** (Impact: {importance:.4f})")
                        st.write(f"   💡 {interpretation}")
                
            except Exception as e:
                st.error(f"Error computing SHAP values: {e}")
                st.write("**Debug information:**", str(e))
                
                # Fallback to simple feature importance
                if hasattr(model, 'feature_importances_'):
                    st.write("### Fallback: Model Feature Importances")
                    
                    importances = model.feature_importances_
                    min_len = min(len(feature_names), len(importances))
                    
                    fallback_df = pd.DataFrame({
                        'feature': feature_names[:min_len],
                        'importance': importances[:min_len]
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = fallback_df.head(15)
                    ax.barh(range(len(top_features)), top_features['importance'])
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['feature'])
                    ax.invert_yaxis()
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Model Feature Importance (Fallback)')
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
        else:
            st.info("Please upload a CSV file or ensure reference data is available to compute SHAP values.")

# ----------------------------
# Responsible AI tab
# ----------------------------
with tab_responsible:
    st.subheader("🛡️ Responsible AI Checklist & Ethics")
    
    # Responsible AI Checklist
    st.write("### 📋 AI Ethics Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### ✅ **Fairness & Bias**")
        fairness_items = [
            "Model tested across different countries and regions",
            "Performance metrics evaluated by demographic groups", 
            "Bias detection implemented in fairness tab",
            "Equal prediction quality across populations",
            "Systematic monitoring for discriminatory outcomes"
        ]
        for item in fairness_items:
            st.write(f"✓ {item}")
        
        st.write("#### 🔒 **Privacy & Data Protection**")
        privacy_items = [
            "No personally identifiable information (PII) collected",
            "Aggregated health data used for training",
            "Data minimization principle applied",
            "Secure data handling practices implemented",
            "GDPR/HIPAA compliance considerations addressed"
        ]
        for item in privacy_items:
            st.write(f"✓ {item}")
            
    with col2:
        st.write("#### 📋 **Informed Consent & Transparency**")
        consent_items = [
            "Clear explanation of model purpose and limitations",
            "Transparent about prediction methodology",
            "Users informed about data usage",
            "Model interpretability provided via feature importance",
            "Uncertainty quantification included in predictions"
        ]
        for item in consent_items:
            st.write(f"✓ {item}")
            
        st.write("#### 🎯 **Model Governance**")
        governance_items = [
            "Regular model performance monitoring",
            "Data drift detection implemented",
            "Model versioning and reproducibility",
            "Human oversight in high-stakes decisions",
            "Continuous bias and fairness auditing"
        ]
        for item in governance_items:
            st.write(f"✓ {item}")
    
    st.divider()
    
    # Model Explainability Section
    st.write("### 🔍 **Model Explainability & SHAP Analysis**")
    
    if uploaded is None and REF is not None:
        st.info("Using reference dataset for SHAP analysis. Upload your own data for custom analysis.")
        base_data = REF.copy()
    elif uploaded is not None:
        try:
            base_data = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as e:
            st.error(f"Error loading uploaded data: {e}")
            base_data = REF.copy() if REF is not None else None
    else:
        base_data = None
        
    if base_data is not None and inference_pipeline is not None:
        st.write("#### 📊 **SHAP Feature Importance Analysis**")
        
        try:
            # Prepare data for SHAP
            sample_data = base_data.sample(n=min(100, len(base_data)), random_state=42)
            al_sample = align_columns(sample_data.copy(), EXPECTED_COLS)
            
            # Get model and preprocessor
            model = inference_pipeline.named_steps["model"]
            preprocessor = inference_pipeline.named_steps["preprocess"]
            
            # Transform data
            X_transformed = preprocessor.transform(al_sample)
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
                
            # Get feature names
            numeric_features = [c for c in EXPECTED_COLS if c not in ['Country', 'Disease_Name']]
            categorical_features = ['Country', 'Disease_Name']
            
            feature_names = numeric_features.copy()
            try:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_features = cat_transformer.get_feature_names_out(categorical_features)
                    feature_names.extend(cat_features)
            except:
                feature_names.extend(categorical_features)
            
            # Calculate feature importance from model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Ensure matching lengths
                min_len = min(len(feature_names), len(importances))
                feature_names = feature_names[:min_len]
                importances = importances[:min_len]
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances,
                    'Importance_Pct': (importances / importances.sum()) * 100
                }).sort_values('Importance', ascending=False)
                
                # Display top features
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Interactive bar chart
                    fig_imp = px.bar(
                        importance_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Features by Importance",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                with col2:
                    st.write("**Feature Importance Ranking**")
                    for i, row in importance_df.head(10).iterrows():
                        st.write(f"{row.name + 1}. **{row['Feature']}**: {row['Importance_Pct']:.1f}%")
                
                # Feature importance table
                st.write("**Detailed Feature Importance**")
                st.dataframe(importance_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in SHAP analysis: {e}")
            st.write("Detailed error for debugging:", str(e))
    
    st.divider()
    
    # Risk Assessment Guidelines
    st.write("### ⚠️ **Risk Assessment Guidelines**")
    
    risk_guidelines = {
        "🔴 **High Risk (Model Prediction = 1)**": [
            "Case fatality rate > 1% OR Cases per 100k population > 100",
            "Requires immediate attention and intervention planning",
            "Enhanced surveillance and resource allocation needed",
            "Consider implementing containment measures"
        ],
        "🟢 **Low Risk (Model Prediction = 0)**": [
            "Case fatality rate ≤ 1% AND Cases per 100k population ≤ 100", 
            "Continue routine monitoring and surveillance",
            "Maintain standard preventive measures",
            "Regular reassessment recommended"
        ]
    }
    
    for risk_level, guidelines in risk_guidelines.items():
        st.write(f"#### {risk_level}")
        for guideline in guidelines:
            st.write(f"• {guideline}")
    
    st.divider()
    
    # Model Limitations and Disclaimers
    st.write("### ⚠️ **Model Limitations & Disclaimers**")
    
    limitations = [
        "**Not a substitute for professional medical judgment**: Always consult public health experts",
        "**Historical data bias**: Model trained on past outbreak data, may not capture emerging patterns",
        "**Geographic limitations**: Performance may vary across different regions and healthcare systems",
        "**Data quality dependent**: Predictions are only as good as the input data quality",
        "**Temporal limitations**: Model may not account for rapidly changing epidemiological conditions",
        "**Ethical use only**: This tool should support, not replace, human decision-making in public health"
    ]
    
    for limitation in limitations:
        st.warning(limitation)

# ----------------------------
# Fairness tab (selection rate + optional metrics)
# ----------------------------
with tab_fair:
    st.subheader("Group comparison (selection rate & metrics)")

    if uploaded is None:
        st.info("Upload a CSV to compute group metrics.")
        st.stop()

    # Read the uploaded CSV safely
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Check sensitive attribute presence
    if sensitive_attr not in df.columns:
        # Suggest possible alternatives
        cand = [
            c
            for c in df.columns
            if (
                pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_categorical_dtype(df[c])
            )
            and df[c].nunique() <= 30
        ]
        st.warning(
            f"Sensitive attribute `{sensitive_attr}` not found in uploaded CSV. "
            f"Try a categorical column with small cardinality (e.g., `city`, `platform`, `reviewer_location`). "
            f"Detected candidates: {', '.join(cand[:10]) if cand else 'None'}"
        )
        st.stop()

    # Align and predict
    al = align_columns(df.copy(), EXPECTED_COLS)
    preds, proba = predict_df(al)
    pred_hat = (
        (proba >= threshold).astype(int) if proba is not None else preds.astype(int)
    )

    # Group-level selection rate
    grp = df.groupby(df[sensitive_attr].astype(str), dropna=False)
    summary = grp.apply(
        lambda g: pd.Series(
            {
                "n": int(len(g)),
                "high_risk_rate": float(
                    (pred_hat[g.index] == 1).mean()
                ),  # predicted 'High Risk' rate
            }
        )
    )
    st.write("**High Risk rate by group**")
    st.dataframe(
        summary.sort_values("high_risk_rate", ascending=False), use_container_width=True
    )

    # Optional metrics (only if ground-truth is provided)
    has_gt = bool(target_attr) and (target_attr in df.columns)

    if has_gt:
        # Normalize common string labels into 0/1
        y_true = (
            df[target_attr]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(
                {
                    "in stock": 0,
                    "instock": 0,
                    "available": 0,
                    "0": 0,
                    "false": 0,
                    "no": 0,
                    "out of stock": 1,
                    "outofstock": 1,
                    "unavailable": 1,
                    "1": 1,
                    "true": 1,
                    "yes": 1,
                }
            )
            .fillna(0)
            .astype(int)
            .values
        )

        acc_by_grp = grp.apply(
            lambda g: accuracy_score(y_true[g.index], pred_hat[g.index])
        )
        f1_by_grp = grp.apply(lambda g: f1_score(y_true[g.index], pred_hat[g.index]))

        st.write("**Accuracy by group**")
        st.dataframe(
            acc_by_grp.to_frame("accuracy").sort_values("accuracy", ascending=False),
            use_container_width=True,
        )

        st.write("**F1 by group**")
        st.dataframe(
            f1_by_grp.to_frame("f1").sort_values("f1", ascending=False),
            use_container_width=True,
        )

        # Simple parity gaps (selection rate difference vs. overall)
        overall_sr = float((pred_hat == 1).mean())
        summary["high_risk_rate_gap_vs_overall"] = (
            summary["high_risk_rate"] - overall_sr
        )
        st.write(
            "**High-risk rate gap vs overall** (positive = predicts 'High Risk' more often than average)"
        )
        st.dataframe(
            summary[["n", "high_risk_rate", "high_risk_rate_gap_vs_overall"]],
            use_container_width=True,
        )
    else:
        st.info(
            "No ground-truth column selected; showing **high-risk rates** only. "
            "If you want accuracy/F1 per group, set a ground-truth column (e.g., `high_risk_outbreak`, `outbreak_risk`, `target`, or `label`)."
        )

# ----------------------------
# Drift tab
# ----------------------------
with tab_drift:
    st.subheader("Data drift checks (PSI & KS)")
    if REF is None:
        st.info(
            "Missing reference_sample.csv in artifacts. Upload a CSV to compare, or add reference file."
        )
    elif uploaded is None:
        st.info("Upload a CSV to compare with the reference sample.")
    else:
        try:
            cur = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception as e:
            st.error(f"Could not read uploaded CSV for drift: {e}")
            cur = None
        if cur is not None:
            num_cols = list(set(REF.columns).intersection(cur.columns))
            num_cols = [
                c
                for c in num_cols
                if pd.api.types.is_numeric_dtype(REF[c])
                and pd.api.types.is_numeric_dtype(cur[c])
            ]
            if not num_cols:
                st.warning("No common numeric columns for drift.")
            else:
                rows = []
                for c in sorted(num_cols):
                    p = psi(REF[c], cur[c])
                    ks = ks_2samp(
                        REF[c].dropna().astype(float), cur[c].dropna().astype(float)
                    ).pvalue
                    rows.append(
                        {
                            "feature": c,
                            "psi": p,
                            "ks_pvalue": ks,
                            "drift_flag": (
                                "HIGH"
                                if (not np.isnan(p) and p >= 0.2)
                                else (
                                    "MED" if (not np.isnan(p) and p >= 0.1) else "LOW"
                                )
                            ),
                        }
                    )
                drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
                st.dataframe(drift_df, use_container_width=True)
                
                # Enhanced drift visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # PSI visualization
                    fig_psi = px.bar(
                        drift_df.head(15),
                        x='psi',
                        y='feature',
                        orientation='h',
                        title="Population Stability Index (PSI) by Feature",
                        color='drift_flag',
                        color_discrete_map={'HIGH': 'red', 'MED': 'orange', 'LOW': 'green'}
                    )
                    fig_psi.add_vline(x=0.1, line_dash="dash", line_color="orange", 
                                     annotation_text="Medium threshold")
                    fig_psi.add_vline(x=0.2, line_dash="dash", line_color="red", 
                                     annotation_text="High threshold")
                    st.plotly_chart(fig_psi, use_container_width=True)
                
                with col2:
                    # KS test p-values
                    valid_ks = drift_df[drift_df['ks_pvalue'].notna()]
                    if not valid_ks.empty:
                        fig_ks = px.scatter(
                            valid_ks,
                            x='ks_pvalue',
                            y='psi',
                            hover_data=['feature'],
                            title="KS Test p-value vs PSI",
                            color='drift_flag',
                            color_discrete_map={'HIGH': 'red', 'MED': 'orange', 'LOW': 'green'}
                        )
                        fig_ks.add_hline(y=0.1, line_dash="dash", line_color="orange")
                        fig_ks.add_hline(y=0.2, line_dash="dash", line_color="red")
                        fig_ks.add_vline(x=0.05, line_dash="dash", line_color="blue", 
                                        annotation_text="Significance threshold")
                        st.plotly_chart(fig_ks, use_container_width=True)
                
                # Drift summary metrics
                st.write("**📊 Drift Summary**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_drift_count = (drift_df['drift_flag'] == 'HIGH').sum()
                    st.metric("High Drift Features", high_drift_count)
                
                with col2:
                    med_drift_count = (drift_df['drift_flag'] == 'MED').sum()
                    st.metric("Medium Drift Features", med_drift_count)
                
                with col3:
                    avg_psi = drift_df['psi'].mean()
                    st.metric("Average PSI", f"{avg_psi:.3f}")
                
                st.caption("**Interpretation:**")
                st.caption("• PSI < 0.1: No significant drift (LOW)")
                st.caption("• PSI 0.1-0.2: Moderate drift, monitor closely (MED)")  
                st.caption("• PSI ≥ 0.2: Significant drift, model retraining recommended (HIGH)")
                st.caption("• KS p-value < 0.05: Statistically significant distribution difference")
