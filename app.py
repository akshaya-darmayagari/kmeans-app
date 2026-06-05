import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Page configuration
st.set_page_config(
    page_title="K-Means Wine Clustering App",
    page_icon="🍷",
    layout="wide"
)

# Soft Light Periwinkle Theme CSS (Dropdown safe)
st.markdown("""
<style>
/* App Main Container - Soft Light Periwinkle background */
[data-testid="stAppViewContainer"] {
    background-color: #f5f3ff;
}

/* Sidebar background styling - Muted Periwinkle Dark */
[data-testid="stSidebar"] {
    background-color: #ede9fe;
    border-right: 1px solid #ddd6fe;
}

/* Base Headings and Text Color Configuration (Dark Slate for High Contrast) */
h1, h2, h3, h4, h5, h6, .stMarkdown p, label, .stMetricValue, [data-testid="stHeader"] {
    color: #0f172a !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

/* Header container with clean purple-indigo styling */
.hero-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
    padding: 35px;
    border-radius: 16px;
    border: 1px solid #ddd6fe;
    border-bottom: 4px solid #7c3aed;
    box-shadow: 0px 10px 30px rgba(124, 58, 237, 0.05);
    margin-bottom: 30px;
    text-align: center;
}

.hero-banner h1 {
    color: #5b21b6 !important;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero-banner p {
    color: #6d28d9 !important;
    font-size: 15px;
}

/* Styled metric card panels - Clean White Cards with soft purple borders */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #ddd6fe;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Card panels for organizing inputs */
.input-card {
    background-color: #ffffff;
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* DataFrame styling */
.stDataFrame {
    background-color: white !important;
    border-radius: 8px;
}

/* Main action buttons styled in Violet/Indigo gradient */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white !important;
    font-weight: 700;
    border-radius: 8px;
    padding: 12px 30px;
    border: none;
    cursor: pointer;
    box-shadow: 0px 4px 15px rgba(124, 58, 237, 0.2);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6);
    box-shadow: 0px 6px 20px rgba(124, 58, 237, 0.3);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <h1>UNSUPERVISED K-MEANS CLUSTERING</h1>
    <p>Wine Chemical Segmentation System • Optimal Profile Centroid Partitioning</p>
</div>
""", unsafe_allow_html=True)

# Load Model & Scaler
try:
    model = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df = pd.read_csv("data/wine_clustering.csv")
except FileNotFoundError:
    st.error("Model artifacts not found. Please run model_training.py first.")
    st.stop()

# Persistent configurations displayed in the sidebar
with st.sidebar:
    st.markdown("### Clustering Configuration")
    st.info(f"**Target Clusters (K):** {model.n_clusters}")
    st.info(f"**Initialization:** k-means++")
    st.info(f"**Inertia (WCSS):** {model.inertia_:.2f}")

# Top horizontal navigation tabs
tab_dashboard, tab_analytics, tab_calculator = st.tabs([
    "📊 Dataset Overview", 
    "🔎 Cluster Projection & Selection", 
    "🔮 Real-Time Profiler"
])

# ========================================
# TAB 1: DATASET OVERVIEW
# ========================================
with tab_dashboard:
    st.subheader("Structure Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unlabeled Instances", f"{df.shape[0]:,}", help="Chemical profiles")
    col2.metric("Chemical Features", df.shape[1])
    col3.metric("Clustering Groups", model.n_clusters)

    st.markdown("---")
    
    st.subheader("Data Matrix Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical Variance Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ========================================
# TAB 2: CLUSTER PROJECTION & SELECTION
# ========================================
with tab_analytics:
    st.subheader("Cluster Visualization")
    
    col_axis, col_plot = st.columns([1, 2])
    
    with col_axis:
        st.markdown("Choose features to project the high-dimensional clusters onto a 2D scatter plot.")
        x_axis = st.selectbox("X-Axis Feature", df.columns, index=0)
        y_axis = st.selectbox("Y-Axis Feature", df.columns, index=9)
        
        st.markdown("---")
        st.write("""
        **About K-Means Partitioning:**
        * Feature scaling is applied to ensure variables are weighted equally regardless of measurement unit.
        * Points are assigned to clusters based on closest Euclidean distance to calculated centroids.
        """)
        
    with col_plot:
        X_scaled = scaler.transform(df)
        labels = model.predict(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#f5f3ff")
        ax.set_facecolor("#ffffff")
        
        scatter = ax.scatter(
            df[x_axis], df[y_axis], 
            c=labels, cmap="rainbow", 
            edgecolors='k', alpha=0.8, s=60
        )
        ax.set_xlabel(x_axis, color='#0f172a')
        ax.set_ylabel(y_axis, color='#0f172a')
        ax.tick_params(colors='#0f172a')
        ax.legend(*scatter.legend_elements(), title="Predicted Clusters", facecolor="#ffffff", edgecolor="#ddd6fe")
        
        st.pyplot(fig)

# ========================================
# TAB 3: REAL-TIME PROFILER
# ========================================
with tab_calculator:
    st.subheader("Profile New Chemical Sample")
    st.write("Provide the chemical dimensions below to evaluate cluster assignment.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_comp, col_phenols, col_color = st.columns(3)

    with col_comp:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #0f172a !important;'>🧪 Basic Composition</h3>", unsafe_allow_html=True)
        alcohol = st.slider("Alcohol (%)", float(df["alcohol"].min()), float(df["alcohol"].max()), float(df["alcohol"].mean()))
        malic_acid = st.slider("Malic Acid", float(df["malic_acid"].min()), float(df["malic_acid"].max()), float(df["malic_acid"].mean()))
        ash = st.slider("Ash", float(df["ash"].min()), float(df["ash"].max()), float(df["ash"].mean()))
        alcalinity_of_ash = st.slider("Alcalinity of Ash", float(df["alcalinity_of_ash"].min()), float(df["alcalinity_of_ash"].max()), float(df["alcalinity_of_ash"].mean()))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_phenols:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #0f172a !important;'>🧬 Phenols & Minerals</h3>", unsafe_allow_html=True)
        magnesium = st.slider("Magnesium", float(df["magnesium"].min()), float(df["magnesium"].max()), float(df["magnesium"].mean()))
        total_phenols = st.slider("Total Phenols", float(df["total_phenols"].min()), float(df["total_phenols"].max()), float(df["total_phenols"].mean()))
        flavanoids = st.slider("Flavanoids", float(df["flavanoids"].min()), float(df["flavanoids"].max()), float(df["flavanoids"].mean()))
        nonflavanoid_phenols = st.slider("Nonflavanoid Phenols", float(df["nonflavanoid_phenols"].min()), float(df["nonflavanoid_phenols"].max()), float(df["nonflavanoid_phenols"].mean()))
        proanthocyanins = st.slider("Proanthocyanins", float(df["proanthocyanins"].min()), float(df["proanthocyanins"].max()), float(df["proanthocyanins"].mean()))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_color:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #0f172a !important;'>🎨 Color & Intensity</h3>", unsafe_allow_html=True)
        color_intensity = st.slider("Color Intensity", float(df["color_intensity"].min()), float(df["color_intensity"].max()), float(df["color_intensity"].mean()))
        hue = st.slider("Hue", float(df["hue"].min()), float(df["hue"].max()), float(df["hue"].mean()))
        od280 = st.slider("OD280/OD315", float(df["od280/od315_of_diluted_wines"].min()), float(df["od280/od315_of_diluted_wines"].max()), float(df["od280/od315_of_diluted_wines"].mean()))
        proline = st.slider("Proline", float(df["proline"].min()), float(df["proline"].max()), float(df["proline"].mean()))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Process inputs on click
    if st.button("RUN CLUSTER SEGMENTATION", use_container_width=True):
        raw_row = np.array([[
            alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
            total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
            color_intensity, hue, od280, proline
        ]])
        
        scaled_row = scaler.transform(raw_row)
        cluster_assignment = model.predict(scaled_row)[0]

        # Distinct styling based on actual model cluster indices
        if cluster_assignment == 0:
            color_theme = "linear-gradient(90deg, #7c3aed, #6d28d9)"
            status_tag = "CHEMICAL PROFILE GROUP 0"
        elif cluster_assignment == 1:
            color_theme = "linear-gradient(90deg, #ec4899, #db2777)"
            status_tag = "CHEMICAL PROFILE GROUP 1"
        else:
            color_theme = "linear-gradient(90deg, #06b6d4, #0891b2)"
            status_tag = "CHEMICAL PROFILE GROUP 2"

        st.markdown(f"""
        <div style="
            background: {color_theme};
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            font-size: 26px;
            font-weight: 700;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            color: white;">
            Segmentation Result: {status_tag}<br>
            <span style='font-size: 16px; font-weight: 400;'>This sample aligns closest to the mathematical centroid of chemical profile {cluster_assignment}.</span>
        </div>
        """, unsafe_allow_html=True)
        
        