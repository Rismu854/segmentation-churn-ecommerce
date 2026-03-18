import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CONFIG PAGE
# ===========================
st.set_page_config(
    page_title="Segmentation & Churn Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CSS PERSONNALISÉ
# ===========================
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric { background: white; padding: 15px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# ===========================
# CHARGEMENT DES DONNÉES
# ===========================
import os

# Chemin dynamique depuis la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    features  = pd.read_csv(
        os.path.join(BASE_DIR, 'data/processed/features_churn.csv'))
    rfm       = pd.read_csv(
        os.path.join(BASE_DIR, 'data/processed/rfm_clustered.csv'))
    high_risk = pd.read_csv(
        os.path.join(BASE_DIR, 'data/processed/clients_haut_risque.csv'))
    return features, rfm, high_risk

@st.cache_resource
def load_models():
    rf        = joblib.load(
        os.path.join(BASE_DIR, 'models/random_forest_churn.pkl'))
    kmeans    = joblib.load(
        os.path.join(BASE_DIR, 'models/kmeans_model.pkl'))
    shap_vals = np.load(
        os.path.join(BASE_DIR, 'models/shap_values.npy'))
    return rf, kmeans, shap_vals

features, rfm, high_risk = load_data()
rf, kmeans, shap_vals    = load_models()

FEATURE_COLS = [
    'Frequency', 'Monetary', 'Avg_OrderValue',
    'Nb_Produits', 'Avg_Quantity', 'Nb_Jours_Actif',
    'Total_Items', 'Cadence', 'Panier_Moyen'
]

# ===========================
# SIDEBAR NAVIGATION
# ===========================
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
st.sidebar.title("🛒 Churn & Segmentation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🛒 Segmentation Clients",
     "🚨 Churn Prediction", "🔍 Explicabilité SHAP"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Données**")
st.sidebar.markdown(f"Clients analysés : **{len(features):,}**")
st.sidebar.markdown(f"Taux de churn : **{features['Churn'].mean()*100:.1f}%**")
st.sidebar.markdown(f"Clients à risque : **{len(high_risk):,}**")


# ===========================
# PAGE 1 : ACCUEIL
# ===========================
if page == "🏠 Accueil":
    st.title("🛒 Dashboard — Segmentation Client & Churn Prediction")
    st.markdown("### Analyse complète d'un dataset e-commerce (Online Retail UK)")
    st.markdown("---")

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="👥 Clients Total",
            value=f"{len(features):,}",
            delta="Dataset complet"
        )
    with col2:
        churned = features['Churn'].sum()
        st.metric(
            label="🚨 Clients Churnés",
            value=f"{churned:,}",
            delta=f"{churned/len(features)*100:.1f}% du total",
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="🔴 Haut Risque (>70%)",
            value=f"{len(high_risk):,}",
            delta="Intervention urgente",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="🏆 AUC-ROC Modèle",
            value="0.8244",
            delta="Random Forest",
            delta_color="normal"
        )

    st.markdown("---")

    # Deux colonnes : répartition churn + segments
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Répartition Churn / Actifs")
        churn_counts = features['Churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=['Actifs', 'Churnés'],
            color_discrete_sequence=['#2E75B6', '#C55A11'],
            hole=0.4
        )
        fig_pie.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("🎯 Répartition par Segment")
        seg_counts = rfm['Cluster_Name'].value_counts()
        fig_seg = px.bar(
            x=seg_counts.values,
            y=seg_counts.index,
            orientation='h',
            color=seg_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Nombre de clients', 'y': ''}
        )
        fig_seg.update_layout(height=350, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Résumé des performances du modèle")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**🥇 Random Forest**\nAUC-ROC : 0.8244\nF1-Score : 0.60")
    with col_b:
        st.warning("**🥈 Logistic Regression**\nAUC-ROC : 0.8159\nF1-Score : 0.64")
    with col_c:
        st.error("**🥉 XGBoost**\nAUC-ROC : 0.8060\nF1-Score : 0.57")



# ===========================
# PAGE 2 : SEGMENTATION
# ===========================
elif page == "🛒 Segmentation Clients":
    st.title("🛒 Segmentation Clients — Clustering K-Means")
    st.markdown("---")

    # Métriques par segment
    st.subheader("📊 Profil de chaque segment")

    segment_profile = rfm.groupby('Cluster_Name').agg(
        Nb_Clients   = ('CustomerID', 'count'),
        Recency_moy  = ('Recency',    'mean'),
        Freq_moy     = ('Frequency',  'mean'),
        Monetary_moy = ('Monetary',   'mean'),
    ).round(1).reset_index()

    st.dataframe(
        segment_profile.style.background_gradient(
            cmap='Blues', subset=['Nb_Clients', 'Freq_moy', 'Monetary_moy']
        ),
        use_container_width=True
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Distribution des segments")
        seg_data = rfm['Cluster_Name'].value_counts().reset_index()
        seg_data.columns = ['Segment', 'Count']
        fig = px.pie(seg_data, values='Count', names='Segment',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     hole=0.3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Monetary moyen par segment")
        fig_bar = px.bar(
            segment_profile.sort_values('Monetary_moy', ascending=True),
            x='Monetary_moy', y='Cluster_Name',
            orientation='h',
            color='Monetary_moy',
            color_continuous_scale='Blues',
            labels={'Monetary_moy': 'Dépense moyenne (£)',
                    'Cluster_Name': ''}
        )
        fig_bar.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Explorer un segment")

    selected_segment = st.selectbox(
        "Choisir un segment :",
        options=rfm['Cluster_Name'].unique()
    )

    seg_clients = rfm[rfm['Cluster_Name'] == selected_segment][
        ['CustomerID', 'Recency', 'Frequency', 'Monetary',
         'R_Score', 'F_Score', 'M_Score', 'RFM_Score']
    ].sort_values('Monetary', ascending=False)

    st.markdown(f"**{len(seg_clients):,} clients dans ce segment**")
    st.dataframe(seg_clients.head(20), use_container_width=True)



# ===========================
# PAGE 3 : CHURN PREDICTION
# ===========================
elif page == "🚨 Churn Prediction":
    st.title("🚨 Churn Prediction — Random Forest")
    st.markdown("---")

    # KPIs churn
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Haut risque (>70%)",
                  f"{(features['Churn_Score']>0.7).sum():,}",
                  "Intervention urgente")
    with col2:
        st.metric("🟠 Risque moyen (50-70%)",
                  f"{((features['Churn_Score']>0.5) & (features['Churn_Score']<=0.7)).sum():,}",
                  "À surveiller")
    with col3:
        st.metric("🟢 Faible risque (<50%)",
                  f"{(features['Churn_Score']<0.5).sum():,}",
                  "Clients stables")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Distribution des scores de risque")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=features[features['Churn']==0]['Churn_Score'],
            name='Actifs', opacity=0.7,
            marker_color='#2E75B6', nbinsx=40
        ))
        fig_hist.add_trace(go.Histogram(
            x=features[features['Churn']==1]['Churn_Score'],
            name='Churnés', opacity=0.7,
            marker_color='#C55A11', nbinsx=40
        ))
        fig_hist.add_vline(x=0.5, line_dash="dash",
                           line_color="black",
                           annotation_text="Seuil 0.5")
        fig_hist.update_layout(
            barmode='overlay', height=380,
            xaxis_title='Score de risque churn',
            yaxis_title='Nombre de clients'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("🎯 Taux de churn par segment")
        churn_seg = features.merge(
            rfm[['CustomerID', 'Cluster_Name']], on='CustomerID'
        ).groupby('Cluster_Name')['Churn'].mean().reset_index()
        churn_seg.columns = ['Segment', 'Taux_Churn']
        churn_seg['Taux_Churn_%'] = (churn_seg['Taux_Churn']*100).round(1)
        churn_seg = churn_seg.sort_values('Taux_Churn_%', ascending=True)

        fig_churn = px.bar(
            churn_seg, x='Taux_Churn_%', y='Segment',
            orientation='h',
            color='Taux_Churn_%',
            color_continuous_scale='RdYlGn_r',
            labels={'Taux_Churn_%': 'Taux de churn (%)', 'Segment': ''}
        )
        fig_churn.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Liste des clients à haut risque")

    risk_filter = st.slider(
        "Seuil de risque minimum (%)",
        min_value=50, max_value=95, value=70, step=5
    )

    filtered = features[
        features['Churn_Score'] >= risk_filter/100
    ][['CustomerID', 'Frequency', 'Monetary',
       'Nb_Jours_Actif', 'Cadence', 'Churn_Score']].copy()

    filtered['Churn_Score_%'] = (filtered['Churn_Score']*100).round(1)
    filtered = filtered.sort_values('Churn_Score', ascending=False)

    st.markdown(f"**{len(filtered):,} clients avec un risque ≥ {risk_filter}%**")
    st.dataframe(
        filtered.drop('Churn_Score', axis=1).head(50),
        use_container_width=True
    )

    # Bouton export
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter la liste CSV",
        data=csv,
        file_name=f'clients_risque_{risk_filter}pct.csv',
        mime='text/csv'
    )


# ===========================
# PAGE 4 : SHAP
# ===========================
elif page == "🔍 Explicabilité SHAP":
    st.title("🔍 Explicabilité SHAP — Pourquoi ce client va churner ?")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Importance globale des features")
        st.image('../reports/figures/shap_importance.png',
                 use_container_width=True)

    with col2:
        st.subheader("🐝 Impact détaillé (Beeswarm)")
        st.image('../reports/figures/shap_beeswarm.png',
                 use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Analyse individuelle d'un client")

    customer_id = st.selectbox(
        "Choisir un client :",
        options=features['CustomerID'].astype(int).sort_values().tolist()
    )

    client_data = features[
        features['CustomerID'] == customer_id
    ].iloc[0]

    client_idx = features[
        features['CustomerID'] == customer_id
    ].index[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        score = client_data['Churn_Score']
        color = "🔴" if score > 0.7 else "🟠" if score > 0.5 else "🟢"
        st.metric("Score de Churn",
                  f"{color} {score*100:.1f}%")
    with col_b:
        st.metric("Fréquence d'achat",
                  f"{client_data['Frequency']:.0f} commandes")
    with col_c:
        st.metric("Dépense totale",
                  f"£{client_data['Monetary']:,.0f}")

    st.markdown("---")
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.subheader("📉 Client à haut risque")
        st.image('../reports/figures/shap_waterfall_high_risk.png',
                 use_container_width=True)

    with col_w2:
        st.subheader("📈 Client à faible risque")
        st.image('../reports/figures/shap_waterfall_low_risk.png',
                 use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Relations features / Churn")
    st.image('../reports/figures/shap_dependence.png',
             use_container_width=True)