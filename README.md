# 🛒 Segmentation Client & Churn Prediction
### Projet Data Science complet — E-commerce Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-ff4b4b?style=flat-square&logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-0.50.0-blueviolet?style=flat-square)

---

## 🏢 1. Problème Métier

### Contexte
Un retailer e-commerce britannique observe une **perte silencieuse de clients** sans pouvoir l'anticiper ni l'expliquer. L'entreprise dispose d'un historique de 541 909 transactions sur 1 an mais n'exploite pas ces données pour prendre des décisions marketing proactives.

### Problèmes identifiés
- **Manque de visibilité** sur qui sont réellement les clients — tous sont traités de la même façon alors qu'ils ont des comportements très différents
- **Réaction tardive** au churn — l'entreprise ne détecte un départ qu'après qu'il s'est produit, quand il est trop tard pour agir
- **Budget marketing mal alloué** — sans segmentation, les campagnes sont envoyées à tous les clients indistinctement, ce qui génère des coûts inutiles et des messages non pertinents

### Questions métier auxquelles ce projet répond
> *"Qui sont mes clients et comment se comportent-ils ?"*
> *"Quels clients risquent de partir dans les prochaines semaines ?"*
> *"Pourquoi un client spécifique est-il à risque de churn ?"*
> *"Quelles actions marketing prendre pour chaque groupe de clients ?"*

### Impact business attendu
Si l'on considère qu'un client churné représente en moyenne **£668 de revenu perdu** (valeur médiane du dataset), les **320 clients à haut risque** représentent un potentiel de **£213 760 de revenus à sauvegarder** avec des actions de rétention ciblées.

---

## 🗺️ 2. Démarche

### Vue d'ensemble du pipeline

```
Données brutes        Nettoyage          Feature             Modélisation
(541 909 lignes)  →  & Preprocessing → Engineering RFM  →  & Évaluation

```

### Étape 1 — Compréhension des données (EDA)
Avant tout modèle, on a exploré les données pour comprendre leur structure et identifier les anomalies :
- **25% de CustomerID manquants** → supprimés car inexploitables pour la segmentation individuelle
- **Commandes annulées** (préfixe 'C') → retirées car elles ne représentent pas un achat réel
- **Quantités négatives et prix à 0** → retirés car ils faussent les calculs RFM
- **Résultat** : 541 909 lignes brutes → 392 669 lignes propres exploitables

### Étape 2 — Feature Engineering RFM
Plutôt que d'utiliser les données brutes, on a construit 3 métriques métier éprouvées en marketing :
- **Recency** : depuis combien de jours le client a-t-il acheté ? (plus c'est récent, mieux c'est)
- **Frequency** : combien de commandes distinctes a-t-il passées ?
- **Monetary** : combien a-t-il dépensé au total ?

Ces 3 métriques ont ensuite été enrichies avec 6 features supplémentaires : valeur moyenne par commande, nombre de produits distincts, cadence d'achat, durée d'activité, quantité moyenne et panier moyen.

### Étape 3 — Segmentation (Unsupervised Learning)
**Choix de K-Means :** Après transformation logarithmique et standardisation, on a appliqué K-Means. Le nombre optimal K=4 a été déterminé par deux méthodes :
- **Méthode Elbow** : coude clairement visible à K=4
- **Silhouette Score** : rebond à K=4 (0.3375) confirmant la cohérence des clusters

**Pourquoi K=4 et pas K=2 ?** Le Silhouette Score recommandait K=2 mathématiquement, mais K=2 donnait seulement "bons clients vs mauvais clients" — trop simpliste pour un usage marketing réel. K=4 offre des segments actionnables et interprétables.

**DBSCAN en complément :** Utilisé pour détecter les outliers — 64 clients au comportement atypique extrême (dont 2 super-VIP avec des achats >£30 000) que K-Means avait absorbés dans les clusters.

### Étape 4 — Churn Prediction (Supervised Learning)
**Définition du churn :** Un client est considéré churné s'il n'a pas acheté depuis plus de 90 jours (seuil standard en e-commerce B2C).

**Pourquoi exclure Recency des features ?** Recency définit directement le churn (Recency > 90j = churné). L'inclure créerait une **fuite de données** (data leakage) — le modèle apprendrait à tricher plutôt qu'à vraiment prédire.

**Comparaison de 4 modèles :** Logistic Regression (baseline), Random Forest, XGBoost, LightGBM — évalués sur AUC-ROC et F1-Score avec cross-validation à 5 folds.

**Optimisation :** Randomized Search (30 itérations par modèle) pour trouver les meilleurs hyperparamètres.

### Étape 5 — Explicabilité SHAP
Pour rendre le modèle utilisable par une équipe marketing non-technique, chaque prédiction est expliquée :
- **Pourquoi globalement** ce modèle prédit-il le churn ? (Summary Plot)
- **Pourquoi ce client spécifique** est-il à risque ? (Waterfall Plot)
- **Quel est le seuil critique** de chaque feature ? (Dependence Plot)

---

## 📊 3. Résultats

### Segmentation — 4 profils clients distincts

| Segment | Clients | Recency | Frequency | Monetary | Taux Churn |
|---|---|---|---|---|---|
| 🥇 Champions | 713 (16.4%) | 12 jours | 13.8 cmds | £8 088 | 0% |
| 💙 Clients Fidèles | 1 166 (26.9%) | 72 jours | 4.1 cmds | £1 802 | 0% |
| 🆕 Nouveaux Actifs | 837 (19.3%) | 18 jours | 2.2 cmds | £558 | 0% |
| ❌ Clients Perdus | 1 622 (37.4%) | 181 jours | 1.3 cmds | £341 | 100% |

**Qualité :** Variance expliquée par PCA = **93.9%** — les 4 clusters sont très bien séparés visuellement en 2D.

### Churn Prediction — Performances des modèles

| Modèle | AUC Baseline | AUC Optimisé | Gain | F1-Score |
|---|---|---|---|---|
| **XGBoost** ⭐ | 0.8060 | **0.8275** | **+0.0418** | 0.6834 |
| Random Forest | 0.8244 | 0.8264 | +0.0207 | 0.6818 |
| LightGBM | 0.8069 | 0.8264 | +0.0195 | 0.5212 |
| Logistic Regression | 0.8159 | 0.8160 | +0.0001 | 0.6149 |

**Cross-validation (5 folds) :**
```
Random Forest       AUC = 0.7812 ± 0.004  ← Très stable
Logistic Regression AUC = 0.7808 ± 0.013
XGBoost             AUC = 0.7599 ± 0.007
```

**Matrice de confusion (Random Forest sur test set) :**
```
                 Prédit Actif    Prédit Churné
Réel Actif           480              98     → 83% bien classifiés
Réel Churné          123             167     → 57% bien détectés
```

### Explicabilité SHAP — Top features

```
Nb_Jours_Actif  ████████████████████████  0.0921  ← Signal le plus fort
Monetary        ██████████               0.0401
Frequency       ██████████               0.0398
Total_Items     ██████████               0.0389
Cadence         ████████                 0.0298
Nb_Produits     ███████                  0.0289
```

**3 insights actionnables :**
- Un client avec **Nb_Jours_Actif = 0** (achat unique) a >80% de probabilité de churner → déclencher une séquence d'onboarding dès le 1er achat
- Dès la **3ème commande**, le risque churn chute drastiquement → le 2ème achat est le moment critique à sécuriser
- Les **Champions** (Monetary > £5 000) ont 0% de churn → investir dans leur fidélisation VIP

### Répartition du risque churn

| Niveau de risque | Clients | Action recommandée |
|---|---|---|
| 🔴 Haut risque (>70%) | 320 | Remise immédiate -20% + appel commercial |
| 🟠 Risque moyen (50-70%) | 1 112 | Email de réengagement personnalisé |
| 🟢 Faible risque (<50%) | 2 906 | Programme fidélité standard |

---

## ⚠️ 4. Limites du Projet

### Limites des données

**Période courte (1 an)**
Le dataset couvre uniquement décembre 2010 à décembre 2011. Avec une seule année, il est impossible d'analyser les tendances saisonnières inter-annuelles ni de valider si le comportement observé est représentatif sur le long terme. *Online Retail II (2 ans) permettrait de pallier cette limite.*

**Absence de données démographiques**
On ne connaît pas l'âge, le genre, la localisation précise ni le canal d'acquisition des clients. Ces informations enrichiraient considérablement la segmentation et permettraient des campagnes encore plus ciblées.

**Données majoritairement UK**
Plus de 90% des transactions viennent du Royaume-Uni. Le modèle est donc peu généralisable à d'autres marchés sans réentraînement sur des données locales.

### Limites du modèle

**Définition du churn simpliste**
Le seuil de 90 jours est arbitraire et identique pour tous les clients. En réalité, un client qui achète toutes les 2 semaines est "churné" beaucoup plus tôt qu'un client qui achète une fois par an. *Une définition personnalisée par client (ex: 3x l'intervalle moyen d'achat) serait plus précise.*

**Écart Cross-Validation vs Test set**
L'AUC en cross-validation (0.78) est inférieur à l'AUC sur le test set (0.83). Cet écart s'explique par la nature temporelle des données mais indique que la performance réelle en production serait probablement autour de **0.78-0.80** et non 0.83.

**Pas de dimension temporelle**
Le modèle prédit le churn à un instant T sans tenir compte de l'évolution du comportement dans le temps. Un modèle de **Survival Analysis** permettrait de prédire *quand* le client va churner, pas seulement *s'il* va churner.

**Présence de clients B2B**
Le dataset contient un mélange de clients particuliers et de grossistes (achats de 74 000 articles en une commande). Ces profils B2B faussent certaines distributions et rendent la définition du churn moins pertinente pour eux. *Idéalement, il faudrait les segmenter séparément.*

### Limites du dashboard

**Données statiques**
Le dashboard affiche des données figées au moment de l'analyse. En production réelle, il faudrait connecter le dashboard à une base de données en temps réel avec mise à jour automatique des scores de churn.

**Pas de mesure d'impact**
Le projet identifie les clients à risque et recommande des actions mais ne mesure pas leur efficacité réelle. Un système d'A/B testing intégré permettrait de valider l'impact des campagnes de rétention.

---

## 🔜 Améliorations Possibles

- [ ] **Online Retail II** — Étendre l'analyse sur 2 ans pour capturer la saisonnalité
- [ ] **Définition churn personnalisée** — Seuil adaptatif par client basé sur son historique
- [ ] **Survival Analysis** — Prédire *quand* le client va churner
- [ ] **Optuna** — Optimisation bayésienne plus poussée des hyperparamètres
- [ ] **MLflow** — Tracking des expériences et versioning des modèles
- [ ] **FastAPI** — API REST pour intégration en production
- [ ] **Tests unitaires** — Couverture du pipeline de données avec pytest

---

## 🛠️ Stack Technique

| Catégorie | Librairies |
|---|---|
| **Data** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn, plotly |
| **ML Clustering** | scikit-learn (KMeans, DBSCAN, PCA) |
| **ML Classification** | scikit-learn, xgboost, lightgbm |
| **Explicabilité** | shap |
| **Optimisation** | RandomizedSearchCV |
| **Dashboard** | streamlit |
| **Utilitaires** | joblib, openpyxl, imbalanced-learn |

---

## 📁 Structure du Projet

```
segmentation-churn/
├── 📓 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing_RFM.ipynb
│   ├── 03_Clustering_KMeans.ipynb
│   ├── 04_Churn_Prediction.ipynb
│   ├── 05_SHAP_Analysis.ipynb
│   └── 06_Optimization.ipynb
├── 📊 data/processed/
│   ├── features_churn.csv
│   ├── rfm_clustered.csv
│   └── clients_haut_risque.csv
├── 🤖 models/
│   ├── best_model_optimized.pkl
│   ├── kmeans_model.pkl
│   └── shap_values.npy
├── 📈 reports/figures/
├── 🖥️ dashboard/app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/TON_USERNAME/segmentation-churn-ecommerce.git
cd segmentation-churn-ecommerce
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 📥 Dataset

Les données brutes ne sont pas incluses (disponibles publiquement) :
- **UCI ML Repository** : [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

Place le fichier dans : `data/raw/Online Retail.xlsx`

---

## 👨‍💻 Auteur

**RAVOAVY Rismu**
- 💼 LinkedIn : [www.linkedin.com/in/ravoavy-rismu-432543326](https://linkedin.com)
- 🐙 GitHub : [github.com/Rismu854](https://github.com)

---
