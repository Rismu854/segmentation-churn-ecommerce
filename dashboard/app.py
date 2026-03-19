import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================================================
# CONFIG PAGE
# ===================================================
st.set_page_config(
    page_title="Segmentation & Churn Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================
# CHEMINS DYNAMIQUES (compatibles Streamlit Cloud)
# ===================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "features"   : os.path.join(BASE_DIR, "data", "processed", "features_churn.csv"),
    "rfm"        : os.path.join(BASE_DIR, "data", "processed", "rfm_clustered.csv"),
    "high_risk"  : os.path.join(BASE_DIR, "data", "processed", "clients_haut_risque.csv"),
    "rf_model"   : os.path.join(BASE_DIR, "models", "random_forest_churn.pkl"),
    "xgb_model"  : os.path.join(BASE_DIR, "models", "xgboost_churn.pkl"),
    "lgb_model"  : os.path.join(BASE_DIR, "models", "best_model_optimized.pkl"),
    "kmeans"     : os.path.join(BASE_DIR, "models", "kmeans_model.pkl"),
    "scaler"     : os.path.join(BASE_DIR, "models", "scaler.pkl"),
    "shap_vals"  : os.path.join(BASE_DIR, "models", "shap_values.npy"),
    "shap_imp"   : os.path.join(BASE_DIR, "reports", "figures", "shap_importance.png"),
    "shap_bee"   : os.path.join(BASE_DIR, "reports", "figures", "shap_beeswarm.png"),
    "shap_wf_h"  : os.path.join(BASE_DIR, "reports", "figures", "shap_waterfall_high_risk.png"),
    "shap_wf_l"  : os.path.join(BASE_DIR, "reports", "figures", "shap_waterfall_low_risk.png"),
    "shap_dep"   : os.path.join(BASE_DIR, "reports", "figures", "shap_dependence.png"),
    "heatmap"    : os.path.join(BASE_DIR, "reports", "figures", "cluster_heatmap.png"),
    "roc"        : os.path.join(BASE_DIR, "reports", "figures", "roc_curves.png"),
}

# ===================================================
# CSS PERSONNALISÉ
# ===================================================
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #2E75B6;
    }
    .section-header {
        background: linear-gradient(90deg, #1F4E79, #2E75B6);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .risk-high   { background:#FFE5E5; border-left:4px solid #C55A11;
                   padding:10px; border-radius:8px; }
    .risk-medium { background:#FFF3E0; border-left:4px solid #FF9800;
                   padding:10px; border-radius:8px; }
    .risk-low    { background:#E8F5E9; border-left:4px solid #1E7B34;
                   padding:10px; border-radius:8px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ===================================================
# GESTION DES ERREURS — CHARGEMENT DES DONNÉES
# ===================================================
@st.cache_data
def load_data():
    missing = [k for k, v in {
        "features"  : PATHS["features"],
        "rfm"       : PATHS["rfm"],
        "high_risk" : PATHS["high_risk"],
    }.items() if not os.path.exists(v)]

    if missing:
        st.error(f"❌ Fichiers manquants : {', '.join(missing)}\n"
                 f"Vérifie que les fichiers CSV sont bien dans `data/processed/`")
        st.stop()

    features  = pd.read_csv(PATHS["features"])
    rfm       = pd.read_csv(PATHS["rfm"])
    high_risk = pd.read_csv(PATHS["high_risk"])
    return features, rfm, high_risk

@st.cache_resource
def load_models():
    models = {}
    for name, path in [("rf",  PATHS["rf_model"]),
                        ("xgb", PATHS["xgb_model"]),
                        ("lgb", PATHS["lgb_model"])]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"⚠️ Modèle '{name}' non trouvé : {path}")

    shap_vals = np.load(PATHS["shap_vals"]) \
                if os.path.exists(PATHS["shap_vals"]) else None

    return models, shap_vals

# Chargement
try:
    features, rfm, high_risk = load_data()
    models, shap_vals = load_models()
    lgb = models.get("lgb")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

FEATURE_COLS = [
    'Frequency', 'Monetary', 'Avg_OrderValue',
    'Nb_Produits', 'Avg_Quantity', 'Nb_Jours_Actif',
    'Total_Items', 'Cadence', 'Panier_Moyen'
]

X = features[FEATURE_COLS]
y = features['Churn']

# ===================================================
# SIDEBAR
# ===================================================
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
st.sidebar.title("🛒 Churn & Segmentation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil",
     "🛒 Segmentation Clients",
     "🚨 Churn Prediction",
     "📈 Performance Modèles",
     "🔍 Explicabilité SHAP"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Statistiques**")
st.sidebar.metric("Clients analysés",  f"{len(features):,}")
st.sidebar.metric("Taux de churn",     f"{features['Churn'].mean()*100:.1f}%")
st.sidebar.metric("Clients à risque",  f"{len(high_risk):,}")
st.sidebar.metric("AUC-ROC (XGBoost)", "0.8275")

# ===================================================
# PAGE 1 : ACCUEIL
# ===================================================
if page == "🏠 Accueil":
    st.title("🛒 Dashboard — Segmentation Client & Churn Prediction")
    st.markdown("### Analyse complète d'un dataset e-commerce (Online Retail UK — 541k transactions)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Clients Total",     f"{len(features):,}")
    with col2:
        churned = features['Churn'].sum()
        st.metric("🚨 Clients Churnés",
                  f"{churned:,}",
                  delta=f"{churned/len(features)*100:.1f}% du total",
                  delta_color="inverse")
    with col3:
        st.metric("🔴 Haut Risque (>70%)",
                  f"{len(high_risk):,}",
                  delta="Intervention urgente",
                  delta_color="inverse")
    with col4:
        st.metric("🏆 Meilleur AUC-ROC", "0.8275",
                  delta="XGBoost optimisé")

    st.markdown("---")
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
        fig_pie.update_layout(height=350)
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
        fig_seg.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Comparaison des modèles")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.info("**🥇 XGBoost**\nAUC : 0.8275\nF1  : 0.6834")
    with col_b:
        st.info("**🥈 Random Forest**\nAUC : 0.8264\nF1  : 0.6818")
    with col_c:
        st.info("**🥈 LightGBM**\nAUC : 0.8264\nF1  : 0.5212")
    with col_d:
        st.warning("**🥉 Logistic Reg.**\nAUC : 0.8160\nF1  : 0.6149")

# ===================================================
# PAGE 2 : SEGMENTATION
# ===================================================
elif page == "🛒 Segmentation Clients":
    st.title("🛒 Segmentation Clients — Clustering K-Means (K=4)")
    st.markdown("---")

    segment_profile = rfm.groupby('Cluster_Name').agg(
        Nb_Clients   = ('CustomerID', 'count'),
        Recency_moy  = ('Recency',    'mean'),
        Freq_moy     = ('Frequency',  'mean'),
        Monetary_moy = ('Monetary',   'mean'),
    ).round(1).reset_index()
    segment_profile['% Clients'] = (
        segment_profile['Nb_Clients'] / len(rfm) * 100
    ).round(1)

    st.subheader("📋 Profil de chaque segment")
    st.dataframe(
        segment_profile.style.background_gradient(
            cmap='Blues',
            subset=['Nb_Clients', 'Freq_moy', 'Monetary_moy']
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
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Monetary moyen par segment")
        fig_bar = px.bar(
            segment_profile.sort_values('Monetary_moy', ascending=True),
            x='Monetary_moy', y='Cluster_Name',
            orientation='h',
            color='Monetary_moy',
            color_continuous_scale='Blues',
            text='Monetary_moy',
            labels={'Monetary_moy': 'Dépense moyenne (£)', 'Cluster_Name': ''}
        )
        fig_bar.update_traces(texttemplate='£%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("🌐 Visualisation interactive — Recency vs Monetary")
    fig_scatter = px.scatter(
        rfm,
        x='Recency', y='Monetary',
        color='Cluster_Name',
        size='Frequency',
        hover_data=['CustomerID', 'Frequency', 'Monetary', 'Recency'],
        title='Clients par Recency vs Monetary (taille = Frequency)',
        color_discrete_sequence=px.colors.qualitative.Set2,
        template='plotly_white'
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Explorer un segment")
    selected_segment = st.selectbox(
        "Choisir un segment :",
        options=sorted(rfm['Cluster_Name'].unique())
    )
    seg_clients = rfm[rfm['Cluster_Name'] == selected_segment][
        ['CustomerID', 'Recency', 'Frequency', 'Monetary',
         'R_Score', 'F_Score', 'M_Score', 'RFM_Score']
    ].sort_values('Monetary', ascending=False)

    st.markdown(f"**{len(seg_clients):,} clients dans ce segment**")
    st.dataframe(seg_clients.head(20), use_container_width=True)

# ===================================================
# PAGE 3 : CHURN PREDICTION
# ===================================================
elif page == "🚨 Churn Prediction":
    st.title("🚨 Churn Prediction — Modèles ML")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        nb_high = (features['Churn_Score'] > 0.7).sum()
        st.metric("🔴 Haut risque (>70%)", f"{nb_high:,}", "Intervention urgente")
    with col2:
        nb_med = ((features['Churn_Score'] > 0.5) &
                  (features['Churn_Score'] <= 0.7)).sum()
        st.metric("🟠 Risque moyen (50-70%)", f"{nb_med:,}", "À surveiller")
    with col3:
        nb_low = (features['Churn_Score'] < 0.5).sum()
        st.metric("🟢 Faible risque (<50%)", f"{nb_low:,}", "Clients stables")

    st.markdown("---")

    # Distribution des scores
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
            yaxis_title='Nombre de clients',
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("🎯 Taux de churn par segment")
        if 'Cluster_Name' in features.columns:
            churn_seg = features.groupby('Cluster_Name')['Churn'].mean().reset_index()
        else:
            churn_seg = features.merge(
                rfm[['CustomerID', 'Cluster_Name']], on='CustomerID'
            ).groupby('Cluster_Name')['Churn'].mean().reset_index()

        churn_seg.columns = ['Segment', 'Taux_Churn']
        churn_seg['Taux_%'] = (churn_seg['Taux_Churn'] * 100).round(1)
        churn_seg = churn_seg.sort_values('Taux_%', ascending=True)

        fig_churn = px.bar(
            churn_seg, x='Taux_%', y='Segment',
            orientation='h',
            color='Taux_%',
            color_continuous_scale='RdYlGn_r',
            text='Taux_%',
            labels={'Taux_%': 'Taux de churn (%)', 'Segment': ''}
        )
        fig_churn.update_traces(texttemplate='%{text:.1f}%',
                                textposition='outside')
        fig_churn.update_layout(height=380, coloraxis_showscale=False,
                                template='plotly_white')
        st.plotly_chart(fig_churn, use_container_width=True)

    st.markdown("---")

    # Liste clients à risque
    st.subheader("🚨 Liste des clients à haut risque")
    risk_filter = st.slider(
        "Seuil de risque minimum (%)",
        min_value=50, max_value=95, value=70, step=5
    )

    filtered = features[
        features['Churn_Score'] >= risk_filter / 100
    ][['CustomerID', 'Frequency', 'Monetary',
       'Nb_Jours_Actif', 'Cadence', 'Churn_Score']].copy()
    filtered['Churn_Score_%'] = (filtered['Churn_Score'] * 100).round(1)
    filtered = filtered.drop('Churn_Score', axis=1)\
                       .sort_values('Churn_Score_%', ascending=False)

    st.markdown(f"**{len(filtered):,} clients avec un risque ≥ {risk_filter}%**")
    st.dataframe(filtered.head(50), use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Exporter la liste CSV",
        data=csv,
        file_name=f'clients_risque_{risk_filter}pct.csv',
        mime='text/csv'
    )

    st.markdown("---")

    # Prédiction nouveau client
    st.subheader("🎯 Prédire le risque d'un nouveau client")
    st.markdown("Remplis les caractéristiques du client pour obtenir son score de churn :")

    col1, col2, col3 = st.columns(3)
    with col1:
        freq         = st.number_input("📦 Fréquence (nb commandes)",    1,    200,   3)
        monetary     = st.number_input("💰 Dépense totale (£)",          10, 100000, 500)
        avg_order    = st.number_input("🧾 Valeur moyenne commande (£)", 5,   50000, 150)
    with col2:
        nb_produits  = st.number_input("🏷️ Nb produits distincts",      1,   1000,   20)
        avg_quantity = st.number_input("📦 Quantité moyenne",             1,   1000,   10)
        nb_jours     = st.number_input("📅 Jours actif (1er→dernier)",   0,    400,   90)
    with col3:
        total_items  = st.number_input("🛒 Total articles achetés",      1,  50000,  200)
        cadence      = st.number_input("⚡ Cadence (cmds/jour)",       0.01,   20.0, 0.05)
        panier_moy   = st.number_input("💳 Panier moyen (£)",             5,  50000,  150)

    if st.button("🔮 Prédire le risque de churn", type="primary"):
        if lgb is None:
            st.error("❌ Modèle XGBoost non disponible.")
        else:
            client_input = np.array([[
                freq, monetary, avg_order, nb_produits,
                avg_quantity, nb_jours, total_items,
                cadence, panier_moy
            ]])
            proba = lgb.predict_proba(client_input)[0][1]

            st.markdown("---")
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                if proba > 0.7:
                    st.markdown(f"""
                    <div class='risk-high'>
                    <h2>🔴 {proba*100:.1f}%</h2>
                    <b>HAUT RISQUE DE CHURN</b>
                    </div>""", unsafe_allow_html=True)
                elif proba > 0.5:
                    st.markdown(f"""
                    <div class='risk-medium'>
                    <h2>🟠 {proba*100:.1f}%</h2>
                    <b>RISQUE MOYEN DE CHURN</b>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='risk-low'>
                    <h2>🟢 {proba*100:.1f}%</h2>
                    <b>FAIBLE RISQUE DE CHURN</b>
                    </div>""", unsafe_allow_html=True)

            with col_res2:
                # Jauge Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={'text': "Score de Risque Churn (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#C55A11" if proba > 0.7
                                else "#FF9800" if proba > 0.5
                                else "#1E7B34"},
                        'steps': [
                            {'range': [0, 50],   'color': "#E8F5E9"},
                            {'range': [50, 70],  'color': "#FFF3E0"},
                            {'range': [70, 100], 'color': "#FFE5E5"},
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("**💡 Action recommandée :**")
            if proba > 0.7:
                st.error("Contacter immédiatement avec une offre de rétention "
                         "(-20% sur la prochaine commande + appel commercial)")
            elif proba > 0.5:
                st.warning("Envoyer un email de réengagement personnalisé "
                           "avec une recommandation produit ciblée")
            else:
                st.success("Maintenir le programme de fidélité standard. "
                           "Ce client est stable.")

# ===================================================
# PAGE 4 : PERFORMANCE MODÈLES
# ===================================================
elif page == "📈 Performance Modèles":
    st.title("📈 Performance des Modèles ML")
    st.markdown("---")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (confusion_matrix, roc_curve,
                                 roc_auc_score, f1_score)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Courbe ROC interactive ──────────────────────
    st.subheader("📉 Courbes ROC — Comparaison interactive")

    fig_roc = go.Figure()
    colors_roc = {
        'Random Forest': '#1E7B34',
        'XGBoost'      : '#C55A11',
        'LightGBM'     : '#7030A0'
    }

    for name, key in [('Random Forest', 'rf'),
                      ('XGBoost',       'xgb'),
                      ('LightGBM',      'lgb')]:
        model = models.get(key)
        if model is None:
            continue
        try:
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC={auc:.4f})',
                line=dict(color=colors_roc[name], width=2.5)
            ))
        except Exception as e:
            st.warning(f"⚠️ Erreur pour {name} : {e}")

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Aléatoire (AUC=0.50)',
        line=dict(color='gray', width=1.5, dash='dash')
    ))
    fig_roc.update_layout(
        title='Courbes ROC — Comparaison des modèles',
        xaxis_title='Taux Faux Positifs (FPR)',
        yaxis_title='Taux Vrais Positifs (TPR)',
        template='plotly_white',
        height=450,
        legend=dict(x=0.6, y=0.1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Matrices de confusion ────────────────────────
    st.subheader("🔲 Matrices de Confusion")

    model_choice = st.selectbox(
        "Choisir un modèle :",
        options=['Random Forest', 'XGBoost', 'LightGBM']
    )

    model_key_map = {
        'Random Forest': 'rf',
        'XGBoost'      : 'xgb',
        'LightGBM'     : 'lgb'
    }
    cmap_map = {
        'Random Forest': 'Blues',
        'XGBoost'      : 'Oranges',
        'LightGBM'     : 'Greens'
    }

    selected_model = models.get(model_key_map[model_choice])

    if selected_model is None:
        st.error(f"❌ Modèle {model_choice} non disponible.")
    else:
        try:
            y_pred  = selected_model.predict(X_test)
            y_proba = selected_model.predict_proba(X_test)[:, 1]
            cm      = confusion_matrix(y_test, y_pred)
            auc     = roc_auc_score(y_test, y_proba)
            f1      = f1_score(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            col_cm, col_stats = st.columns([1, 1])

            with col_cm:
                fig_cm, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d',
                            cmap=cmap_map[model_choice],
                            xticklabels=['Actif', 'Churné'],
                            yticklabels=['Actif', 'Churné'],
                            ax=ax, annot_kws={'size': 16})
                ax.set_title(f'Matrice de Confusion\n{model_choice}',
                             fontsize=13, fontweight='bold')
                ax.set_ylabel('Réel')
                ax.set_xlabel('Prédit')
                plt.tight_layout()
                st.pyplot(fig_cm)

            with col_stats:
                st.markdown("### 📊 Métriques détaillées")
                st.markdown(f"""
                | Métrique | Valeur |
                |---|---|
                | **AUC-ROC** | {auc:.4f} |
                | **F1-Score** | {f1:.4f} |
                | **Vrais Négatifs (TN)** | {tn:,} |
                | **Faux Positifs (FP)** | {fp:,} |
                | **Faux Négatifs (FN)** | {fn:,} |
                | **Vrais Positifs (TP)** | {tp:,} |
                | **Précision (Churnés)** | {tp/(tp+fp)*100:.1f}% |
                | **Rappel (Churnés)** | {tp/(tp+fn)*100:.1f}% |
                """)

                st.markdown("---")
                st.markdown("### 💡 Interprétation")
                st.info(
                    f"**{tp}** churnés correctement détectés sur **{tp+fn}** réels "
                    f"({tp/(tp+fn)*100:.1f}% de rappel)\n\n"
                    f"**{fp}** faux positifs — clients actifs alertés à tort "
                    f"(coût faible : campagne inutile)\n\n"
                    f"**{fn}** faux négatifs — churnés non détectés "
                    f"(coût élevé : revenus perdus)"
                )
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

    st.markdown("---")

    # ── Feature Importance ──────────────────────────
    st.subheader("🏆 Feature Importance du modèle")

    fi_model_choice = st.selectbox(
        "Modèle pour la feature importance :",
        options=['Random Forest', 'XGBoost', 'LightGBM'],
        key='fi_select'
    )

    fi_model = models.get(model_key_map[fi_model_choice])

    if fi_model is not None and hasattr(fi_model, 'feature_importances_'):
        importances = fi_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature'    : FEATURE_COLS,
            'Importance' : importances
        }).sort_values('Importance', ascending=True)

        fig_fi = px.bar(
            fi_df,
            x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            text='Importance',
            title=f'Feature Importance — {fi_model_choice}',
            labels={'Importance': 'Importance', 'Feature': ''},
            template='plotly_white'
        )
        fig_fi.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        fig_fi.update_layout(
            height=420,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Top 3 features
        top3 = fi_df.sort_values('Importance', ascending=False).head(3)
        st.markdown("**🎯 Top 3 features les plus importantes :**")
        for i, (_, row) in enumerate(top3.iterrows()):
            medals = ["🥇", "🥈", "🥉"]
            st.markdown(
                f"{medals[i]} **{row['Feature']}** — "
                f"importance : `{row['Importance']:.4f}`"
            )
    else:
        st.warning(f"⚠️ Feature importance non disponible pour {fi_model_choice}")

    st.markdown("---")

    # ── Tableau récapitulatif ───────────────────────
    st.subheader("📋 Tableau récapitulatif — Tous les modèles")
    summary = []
    for name, key in [('Random Forest', 'rf'),
                      ('XGBoost',       'xgb'),
                      ('LightGBM',      'lgb')]:
        m = models.get(key)
        if m is None:
            continue
        try:
            yp    = m.predict(X_test)
            yprob = m.predict_proba(X_test)[:, 1]
            tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
            summary.append({
                'Modèle'   : name,
                'AUC-ROC'  : round(roc_auc_score(y_test, yprob), 4),
                'F1-Score' : round(f1_score(y_test, yp), 4),
                'Précision': f"{tp/(tp+fp)*100:.1f}%",
                'Rappel'   : f"{tp/(tp+fn)*100:.1f}%",
                'TP'       : tp,
                'FP'       : fp,
                'FN'       : fn,
                'TN'       : tn,
            })
        except Exception:
            pass

    if summary:
        df_summary = pd.DataFrame(summary)
        st.dataframe(
            df_summary.style.highlight_max(
                subset=['AUC-ROC', 'F1-Score'],
                color='#D6E4F0'
            ),
            use_container_width=True
        )

# ===================================================
# PAGE 5 : SHAP
# ===================================================
elif page == "🔍 Explicabilité SHAP":
    st.title("🔍 Explicabilité SHAP — Pourquoi ce client va churner ?")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Importance globale (SHAP)")
        if os.path.exists(PATHS["shap_imp"]):
            st.image(PATHS["shap_imp"], use_container_width=True)
        else:
            st.warning("⚠️ Image non trouvée : shap_importance.png")

    with col2:
        st.subheader("🐝 Beeswarm Plot")
        if os.path.exists(PATHS["shap_bee"]):
            st.image(PATHS["shap_bee"], use_container_width=True)
        else:
            st.warning("⚠️ Image non trouvée : shap_beeswarm.png")

    st.markdown("---")
    st.subheader("📊 Feature Importance SHAP (interactive)")

    if shap_vals is not None:
        mean_shap = pd.DataFrame({
            'Feature'    : FEATURE_COLS,
            'Importance' : np.abs(shap_vals).mean(axis=0)
        }).sort_values('Importance', ascending=True)

        fig_shap = px.bar(
            mean_shap,
            x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Purples',
            text='Importance',
            title='Importance moyenne SHAP (|valeur SHAP|)',
            template='plotly_white'
        )
        fig_shap.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        fig_shap.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.warning("⚠️ Valeurs SHAP non disponibles (shap_values.npy manquant)")

    st.markdown("---")
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.subheader("🔴 Client à haut risque")
        if os.path.exists(PATHS["shap_wf_h"]):
            st.image(PATHS["shap_wf_h"], use_container_width=True)
        else:
            st.warning("⚠️ Image non trouvée")

    with col_w2:
        st.subheader("🟢 Client à faible risque")
        if os.path.exists(PATHS["shap_wf_l"]):
            st.image(PATHS["shap_wf_l"], use_container_width=True)
        else:
            st.warning("⚠️ Image non trouvée")

    st.markdown("---")
    st.subheader("📈 Dependence Plots")
    if os.path.exists(PATHS["shap_dep"]):
        st.image(PATHS["shap_dep"], use_container_width=True)
    else:
        st.warning("⚠️ Image non trouvée : shap_dependence.png")

    st.markdown("---")
    st.subheader("🔍 Analyse individuelle d'un client")

    customer_id = st.selectbox(
        "Choisir un client :",
        options=sorted(features['CustomerID'].astype(int).tolist())
    )

    client_row = features[features['CustomerID'] == customer_id].iloc[0]
    score      = client_row.get('Churn_Score', 0)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        color = "🔴" if score > 0.7 else "🟠" if score > 0.5 else "🟢"
        st.metric("Score de Churn", f"{color} {score*100:.1f}%")
    with col_b:
        st.metric("Fréquence", f"{client_row['Frequency']:.0f} commandes")
    with col_c:
        st.metric("Dépense totale", f"£{client_row['Monetary']:,.0f}")

    col_d, col_e = st.columns(2)
    with col_d:
        st.metric("Jours actif", f"{client_row['Nb_Jours_Actif']:.0f} jours")
    with col_e:
        st.metric("Panier moyen", f"£{client_row['Panier_Moyen']:,.0f}")