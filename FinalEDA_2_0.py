#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[138]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import shap
from dateutil import parser
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, silhouette_score, precision_recall_fscore_support, auc
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[139]:


df = pd.read_csv("../Downloads/deku_cleaned_final.csv")


# In[140]:


print(df.shape)
print(df.columns)


# In[141]:


df.info()


# In[142]:


df.describe


# Missing Values?

# In[143]:


# Missing values per column
df.isnull().sum().sort_values(ascending=False)


# Basic Statistics

# In[144]:


# Summary stats for numeric columns
print(df.describe())

# Unique Value Count by Column
df.select_dtypes(include='object').nunique()


# In[145]:


# Total non-missing counts per column
non_missing_counts = df.notnull().sum()

# Maximum possible record count
max_count = len(df)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=non_missing_counts.index, y=non_missing_counts.values, palette="coolwarm")

# Add horizontal line for max records
plt.axhline(y=max_count, color="green", linestyle="--", linewidth=2, label=f"Max Records ({max_count})")

# Set y-axis limit
plt.ylim(0, 30000)

plt.title("Non-Missing Value Counts per Feature", fontsize=14, weight='bold')
plt.ylabel("Number of Records")
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# In[146]:


import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# -----------------------------
# CONFIG
# -----------------------------
TOP_N = 10
FIGSIZE = (16, 5)   # one “slide”-like wide figure
BAR_COLOR = "#3b82f6"   # bright blue
TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 11

# -----------------------------
# LOAD
# -----------------------------

# Columns from your dataset
col_genre = "genre_primary_grouped"
col_platform = "platform_cleaned"
col_language = "languages"

# -----------------------------
# PREP
# -----------------------------
def top_counts(series, top_n=10):
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .value_counts()
        .head(top_n)
        .sort_values(ascending=True)  # sort ascending for a horizontal bar chart
    )

genre_counts = top_counts(df[col_genre], TOP_N)
platform_counts = top_counts(df[col_platform], TOP_N)
language_counts = top_counts(df[col_language], TOP_N)

# Helper to wrap long tick labels
def wrap_labels(labels, width=14):
    return ["\n".join(textwrap.wrap(str(lbl), width=width)) for lbl in labels]

# -----------------------------
# PLOT
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True)

plots = [
    (axes[0], genre_counts, "Top Genres"),
    (axes[1], platform_counts, "Top Platforms"),
    (axes[2], language_counts, "Top Languages"),
]

for ax, counts, title in plots:
    ax.barh(wrap_labels(counts.index), counts.values, color=BAR_COLOR)
    ax.set_title(title, fontsize=TITLE_SIZE, weight="bold")
    ax.set_xlabel("Number of Titles", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    # Annotate bars with counts
    for i, v in enumerate(counts.values):
        ax.text(v + max(counts.values)*0.01, i, str(v), va="center", fontsize=TICK_SIZE)

# A single super-title for the slide
fig.suptitle("Dataset Diversity: Genres | Platforms | Languages", fontsize=18, weight="bold", y=1.05)

# Optional: save for your slide
# fig.savefig("diversity_triptych.png", dpi=200, bbox_inches="tight")

plt.show()


# In[147]:


import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib import rcParams

rcParams["figure.figsize"] = (12, 7)
rcParams["savefig.dpi"] = 200

def rounded_box(ax, xy, w, h, label, fontsize=12, weight="normal"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.08")
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fontsize, weight=weight)
    return (x + w/2, y + h/2), (w, h)

def badge(ax, center, text, r=0.6, fontsize=10, weight="bold"):
    cx, cy = center
    circ = Circle((cx, cy), r)
    ax.add_patch(circ)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return (cx, cy)

def arrow(ax, a, b, rad=0.0, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(posA=a, posB=b, arrowstyle=style, mutation_scale=16,
                                 connectionstyle=f"arc3,rad={rad}", linewidth=lw))

# Canvas
fig, ax = plt.subplots()
ax.axis("off")
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)

# Title
ax.text(0.5, 8.6, "Scalability • Security • Compliance", fontsize=16, weight="bold", ha="left")

# Left: Data source
src_center, _ = rounded_box(ax, (0.7, 5.5), 3.0, 1.2, "Dekudeals\n(Web Scrape)\nNo PII", fontsize=11)
ax.text(2.2, 4.9, "Data Ingest\n(ETL/QA)", ha="center", fontsize=10)

# Center: Streamlit App container
streamlit_center, _ = rounded_box(ax, (5.0, 5.2), 3.2, 1.6, "Streamlit App\n(Containerized)", fontsize=12, weight="bold")

# Right: Hugging Face Spaces hosting
hf_center, _ = rounded_box(ax, (9.6, 5.2), 3.5, 1.6, "Hugging Face Spaces\n(Hosting)", fontsize=12, weight="bold")

# Autoscaling arrows above HF
ax.text(11.35, 7.2, "Auto-Scaling", fontsize=11, ha="center")
arrow(ax, (11.35, 6.9), (11.35, 6.0), rad=0.0, lw=2.0)
arrow(ax, (10.6, 6.8), (11.35, 6.0), rad=0.2, lw=1.6)
arrow(ax, (12.1, 6.8), (11.35, 6.0), rad=-0.2, lw=1.6)

# Flow arrows
arrow(ax, src_center, streamlit_center, rad=0.0)
arrow(ax, streamlit_center, hf_center, rad=0.0)

# Bottom badges
sec = badge(ax, (6.2, 2.4), "Security\nRBAC + TLS", r=0.85, fontsize=10)
comp = badge(ax, (8.5, 2.4), "Compliance\nGDPR / No PII", r=0.85, fontsize=10)
obs = badge(ax, (10.8, 2.4), "Observability\nLogs / Monitors", r=0.85, fontsize=10)

# Connect badges
arrow(ax, streamlit_center, sec, rad=0.0, lw=1.4)
arrow(ax, hf_center, comp, rad=0.0, lw=1.4)
arrow(ax, hf_center, obs, rad=0.1, lw=1.4)

# Side notes
ax.text(5.0, 3.4, "Container\nIsolation", ha="center", fontsize=10)
ax.text(9.6, 3.4, "Infra as Code\nReproducible Builds", ha="center", fontsize=10)

# Footer legend
ax.text(0.7, 0.8,
        "Scalability: Hugging Face auto-scaling for traffic spikes\n"
        "Security: Encrypted transport, Role-Based Access Control\n"
        "Compliance: No PII; GDPR-aligned practices (data minimization, portability)",
        fontsize=9, ha="left", va="center")

plt.tight_layout()
# Save if desired:
# fig.savefig("scalability_security_compliance.png", bbox_inches="tight", dpi=200)
plt.show()


# In[148]:


plt.figure(figsize=(10, 5))
df['msrp'].hist(bins=50)
plt.title("MSRP Distribution")
plt.xlabel("MSRP ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[149]:


# View high MSRP values
df[df['msrp'] > 100].sort_values(by='msrp', ascending=False).head()


# General Distributions

# In[150]:


# MSRP
sns.histplot(df["msrp"], bins=40, kde=True)
plt.title("Distribution of MSRP Prices")
plt.xlabel("MSRP ($)")
plt.show()

# Game Age
sns.histplot(df["game_age"], bins=15, kde=True)
plt.title("Age of Game Distribution")
plt.xlabel("Years Since Release")
plt.show()

# Platform Distribution
sns.countplot(y="platform_cleaned", data=df, order=df["platform_cleaned"].value_counts().index)
plt.title("Platform Distribution")
plt.show()


# In[151]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Compute stats
avg = df["msrp"].mean()
low, high = 5, 18

# Plot
plt.figure(figsize=(10, 5))
# Histogram
plt.hist(df["msrp"], bins=40, edgecolor="white")
# Shade $5–$18 band
plt.axvspan(low, high, alpha=0.15, label=f"Common band ${low}–${high}")
# Mean line
plt.axvline(avg, linestyle="--", linewidth=2, label=f"Average MSRP = ${avg:.2f}")

plt.title("MSRP Distribution")
plt.xlabel("MSRP ($)")
plt.ylabel("Number of Titles")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# MSRP by Key Variables

# In[152]:


# MSRP by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x="genre_primary_grouped", y="msrp", data=df)
plt.xticks(rotation=45)
plt.title("MSRP by Genre")
plt.show()

# MSRP vs Game Age
sns.scatterplot(x="game_age", y="msrp", hue="is_multiplayer", data=df)
plt.title("MSRP vs Age of Game")
plt.show()


# Multiplayer Breakdown

# In[153]:


# Multiplayer vs Singleplayer Breakdown
sns.countplot(x="is_multiplayer", data=df)
plt.title("Multiplayer (1) vs Single Player (0)")
plt.show()

# Multiplayer by Genre
pd.crosstab(df["genre_primary_grouped"], df["is_multiplayer"]).plot(kind='bar', stacked=True, figsize=(12,6))
plt.title("Multiplayer Split by Genre")
plt.show()


# In[154]:


import matplotlib.pyplot as plt

# Example values based on your stat
multiplayer_pct = 31.6
singleplayer_pct = 100 - multiplayer_pct

labels = ["Multiplayer", "Single Player"]
sizes = [multiplayer_pct, singleplayer_pct]
colors = ["#66c2a5", "#fc8d62"]
explode = (0.05, 0)  # Slightly explode the multiplayer slice

plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    textprops={"fontsize": 12}
)

plt.title("Game Type Split", fontsize=14, weight="bold")
plt.tight_layout()
plt.show()


# Correlation Heatmap

# In[155]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# In[156]:


#  a generic schema of platforms and databases feeding a dashboard from df.





# # Modeling:

# ## Classification - Predicting the Need for a Sequel

# Imports

# In[157]:


# Optional: Enable JS for SHAP plots (especially in Jupyter)
shap.initjs()


# In[158]:


df.info()


# Feature Selection/Encoding

# In[159]:


# Updated selected_features (removes unencoded object columns)
selected_features = [
    'msrp', 'platform_count', 'min_players', 'max_players', 'is_multiplayer',
    'release_year', 'game_age', 'genre_count',
    'mode_handheld', 'mode_docked', 'mode_tabletop',
    'developer_grouped', 'publisher_grouped', 'languages'
] + [col for col in df.columns if col.startswith('genre_') and df[col].dtype != 'object']


# In[160]:


'msrp', 'platform_count', 'min_players', 'max_players', 'is_multiplayer',
'release_year', 'game_age', 'genre_count',
'mode_handheld', 'mode_docked', 'mode_tabletop',
'developer_grouped', 'publisher_grouped', 'languages'


# In[161]:


for col in selected_features:
    dtype = df[col].dtype
    if dtype == 'object':
        print(f"⚠️ Column '{col}' is still object type.")


# Encoding
# 

# In[162]:


categorical_cols = ['developer_grouped', 'publisher_grouped', 'languages']
df_encoded = pd.get_dummies(df[selected_features], columns=categorical_cols, drop_first=True)


# Confirm all columns are numeric
assert df_encoded.select_dtypes(include='object').empty, "Non-numeric columns still exist!"


# Splitting

# In[163]:


# Define features and target
X = df_encoded
y = df['is_sequel']

# Split into train_temp and test (80/20)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Further split train_temp into train and validation (60/20/20 overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.25, stratify=y_train_temp, random_state=42
)


# In[164]:


from scipy.stats import randint, uniform
# ✅ Parameter distributions for RandomizedSearchCV


param_dist = {
    'max_depth': [20, 30, 40, None],             # Keep deeper trees, drop shallow ones
    'min_samples_leaf': randint(1, 4),           # 1–3 leaves (small values were better)
    'min_samples_split': randint(2, 8),          # 2–7 splits (small/moderate values)
    'max_features': ['sqrt', 0.3, 0.5],          # Focus on promising fractions
    'n_estimators': randint(300, 501),           # Higher range since more trees improved AUC
}


# In[165]:


# ✅ Base model with class_weight balanced
rf_base = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# ✅ RandomizedSearchCV
rand_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=150,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2,
    refit=True,
    random_state=42
)


# In[166]:


# ✅ Fit search
rand_search.fit(X_train, y_train)

# ✅ Best model
best_rf = rand_search.best_estimator_
print("Best Hyperparameters:\n", rand_search.best_params_)
print("Best CV AUC Score: {:.4f}".format(rand_search.best_score_))


# In[167]:


# ✅ Function to plot ROC
def plot_roc_curve(model, X_data, y_data, label_set="Validation"):
    y_proba = model.predict_proba(X_data)[:, 1]
    fpr, tpr, _ = roc_curve(y_data, y_proba)
    auc_score = roc_auc_score(y_data, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve ({label_set})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    return auc_score





# In[168]:


# ✅ Validation + Test evaluation
val_auc = plot_roc_curve(best_rf, X_val, y_val, label_set="Validation")
test_auc = plot_roc_curve(best_rf, X_test, y_test, label_set="Test")



# In[169]:


y_pred_test = best_rf.predict(X_test)
print("Test Set Classification Report (Tuned):\n")
print(classification_report(y_test, y_pred_test))


# - True Negatives: 3318 Correctly predicted as NOT SEQUEL games
# - False Positives: 379 Predicted sequel but weren't
# - False Negatives: 318 Missed sequel predictions
# - True Positives: 410 Correctly predicted sequels
# 

# In[170]:


410/(410 + 379)


# In[171]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Display with labels for sequel vs. not sequel
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Sequel (0)", "Sequel (1)"]
)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Tuned Random Forest")
plt.show()


# In[172]:


from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# ✅ Fix duplicate column names
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_val = X_val.loc[:, ~X_val.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]

print(f"X_train shape after deduplication: {X_train.shape}")


# ✅ Refit the model after deduplication so importances align
best_rf.fit(X_train, y_train)

# ✅ Get top N features safely
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 6
top_features = list(X_train.columns[indices[:top_n]])

print("Top features for PDP:", top_features)

# ✅ Plot PDPs
fig, ax = plt.subplots(figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    best_rf,
    X_train,
    features=top_features,
    grid_resolution=20,
    ax=ax
)
plt.tight_layout()
plt.show()



# ## CEO Plots

# In[173]:


# If you don't already have probabilities
y_score = best_rf.predict_proba(X_test)[:, 1]


# In[174]:


# =========================
# 1) CONFUSION MATRIX (counts + % + plain-English labels)
# =========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example data
cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
cm_sum = cm.sum()
cm_pct = cm / cm_sum * 100

labels_display = np.array([["True: No Sequel\nPred: No Sequel",
                            "True: No Sequel\nPred: Sequel"],
                           ["True: Sequel\nPred: No Sequel",
                            "True: Sequel\nPred: Sequel"]])

fig, ax = plt.subplots(figsize=(7, 6))

# ✅ Use stronger blue shades with no white
im = ax.imshow(cm_pct, cmap=plt.cm.Blues_r)

ax.set_title("Prediction Accuracy by Sequel Status", fontsize=14, fontweight="bold", color="white")
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Pred: No Sequel", "Pred: Sequel"], fontsize=12, color="white")
ax.set_yticklabels(["True: No Sequel", "True: Sequel"], fontsize=12, color="white")

# ✅ Add text with outline for readability
for i in range(2):
    for j in range(2):
        text = f"{labels_display[i,j]}\n{cm[i,j]:,} ({cm_pct[i,j]:.1f}%)"
        ax.text(j, i, text,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])

# ✅ Add colorbar with white labels
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

# Black background for contrast
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

plt.tight_layout()
plt.show()


# In[175]:


# =========================
# 2) PRECISION / RECALL / F1 BARS (per class)
# =========================
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred_test, labels=[0,1], zero_division=0
)

metrics = ["Precision", "Recall", "F1-score"]
no_sequel_vals = [prec[0], rec[0], f1[0]]
sequel_vals    = [prec[1], rec[1], f1[1]]

x = np.arange(len(metrics)); width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, no_sequel_vals, width, label="No Sequel", color="#E60012")  # Nintendo red
ax.bar(x + width/2, sequel_vals,    width, label="Sequel",    color="#FFFFFF", edgecolor="#333333")

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score (0–1)")
ax.set_title("Performance by Class (Higher = Better)")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.legend()

for i, v in enumerate(no_sequel_vals):
    ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
for i, v in enumerate(sequel_vals):
    ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()


# In[ ]:


# =========================
# 3) ROC CURVE (with AUC)
# =========================
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Model ROC (AUC = {roc_auc:.2f})", color="#1E88E5")
ax.plot([0,1], [0,1], lw=1, linestyle="--", label="Random Guess", color="#888888")

ax.set_xlabel("False Positive Rate (Costly False Alarms)")
ax.set_ylabel("True Positive Rate (Sequel Hits Captured)")
ax.set_title("ROC Curve – Separating Winners from Non‑Winners")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[176]:


from sklearn.metrics import precision_recall_fscore_support

def eval_at_threshold(th):
    preds = (y_score >= th).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0,1], zero_division=0)
    return {"threshold": th, "sequel_precision": prec[1], "sequel_recall": rec[1], "sequel_f1": f1[1]}

for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
    print(eval_at_threshold(th))


# In[ ]:


# =========================
# 4) (Optional) BUSINESS IMPACT SUMMARY
#     Turn confusion matrix into $$$ for the CEO
# =========================
# Set your assumptions here (example numbers—update to your business case):
REVENUE_PROFITABLE_SEQUEL = 50_000_000   # $ gained if we correctly greenlight a good sequel (TP)
COST_BAD_SEQUEL = 50_000_000            # $ lost if we greenlight a bad sequel (FP)
OPP_COST_MISSED_SEQUEL = 50_000_000     # $ lost if we miss a profitable sequel (FN)
SAVINGS_AVOIDED_BAD = 30_000_000        # $ saved if we correctly reject a bad sequel (TN)

TN, FP = cm[0,0], cm[0,1]
FN, TP = cm[1,0], cm[1,1]

value_tp = TP * REVENUE_PROFITABLE_SEQUEL
value_tn = TN * SAVINGS_AVOIDED_BAD
cost_fp  = FP * COST_BAD_SEQUEL
cost_fn  = FN * OPP_COST_MISSED_SEQUEL

net_value = value_tp + value_tn - cost_fp - cost_fn

labels = ["Correct No-Sequel (TN)\nSavings",
          "Correct Sequel (TP)\nRevenue",
          "Wrong Sequel (FP)\nCost",
          "Missed Sequel (FN)\nOpp. Cost"]
values = [value_tn, value_tp, -cost_fp, -cost_fn]
colors = ["#4CAF50", "#1E88E5", "#E53935", "#FB8C00"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, values, color=colors)

ax.set_title("Business Impact Summary (Assumption-Driven)")
ax.set_ylabel("Estimated $ Impact")
ax.axhline(0, color="black", linewidth=1)

# Value annotations
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + (0.02*max(values) if h>=0 else 0.02*max(values)),
            f"${h:,.0f}", ha="center", va="bottom" if h>=0 else "top", fontsize=10, rotation=0)

# Net value callout
ax.text(0.5, 0.95, f"Net Estimated Value: ${net_value:,.0f}",
        ha="center", va="center", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()


# ## Cluster Analysis - Tentative

# ### Clustering Games by Revenue Potential

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


rev_features = [
    'msrp',
    'platform_count',
    'release_year',
    'game_age',
    'is_multiplayer',
    'genre_count'
] + [col for col in df_encoded.columns if col.startswith('genre_')]


# Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Subset features
X_rev = df_encoded[rev_features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rev)

# PCA — keep enough components to explain ~90% variance
pca = PCA(n_components=0.9, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced from {X_scaled.shape[1]} → {X_pca.shape[1]} components")

# ---------------------------------
# 2️⃣ Elbow Method to find optimal K
# ---------------------------------
K_range = range(2, 11)
wcss = []
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))

# Plot Elbow
plt.figure(figsize=(8,4))
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Elbow Method (PCA Features)')
plt.grid(True)
plt.show()
# ---------------------------------
# 3️⃣ Silhouette Score for optimal K
# ---------------------------------
# Print Silhouette
for k, score in zip(K_range, sil_scores):
    print(f"K={k} → Silhouette: {score:.3f}")
# ---------------------------------
# 4️⃣ Fit Final Model with Chosen K
# (Pick K from Elbow/Silhouette results)
# ---------------------------------
# Fit best K (example: k=3)
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)



# ---------------------------------
# 5️⃣ Visualize Clusters on Two Features
# (Pick any 2 numeric columns for plotting)
# ---------------------------------
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=250, c='red', marker='X', label='Centroids')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'KMeans Clusters (K={best_k}) - PCA Projection')
plt.legend()
plt.show()


# # Business Use Cases

# ## Ignore

# Assumptions

# In[ ]:


# --- inputs you already have ---
# y_test, y_score (best_rf.predict_proba(X_test)[:,1])

# --- set realistic unit economics + decision volume ---
# Adjust these!
VAL_TP = 35_000_000      # profit if we greenlight a truly good sequel
VAL_TN = 5_000_000       # savings if we correctly reject a bad sequel (lower than $30M to avoid swamping)
COST_FP = 25_000_000     # loss if we greenlight a bad sequel
COST_FN = 35_000_000     # opp. cost if we miss a good sequel
DECISIONS_PER_YEAR = 15 # how many sequel decisions we actually make in a year


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve


def dollars_from_rates(tn_rate, fp_rate, fn_rate, tp_rate,
                       n_decisions=DECISIONS_PER_YEAR,
                       val_tp=VAL_TP, val_tn=VAL_TN, cost_fp=COST_FP, cost_fn=COST_FN):
    TN = tn_rate * n_decisions
    FP = fp_rate * n_decisions
    FN = fn_rate * n_decisions
    TP = tp_rate * n_decisions
    value_tp = TP * val_tp
    value_tn = TN * val_tn
    cost_fp  = FP * cost_fp
    cost_fn  = FN * cost_fn
    net = value_tp + value_tn - cost_fp - cost_fn
    return net, value_tn, value_tp, -cost_fp, -cost_fn

def cm_rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    return tn/total, fp/total, fn/total, tp/total, cm

# --- sweep thresholds, choose the EV-maximizing one ---
ths = np.linspace(0.1, 0.9, 17)
evs, details = [], []

for th in ths:
    y_pred = (y_score >= th).astype(int)
    tn_r, fp_r, fn_r, tp_r, cm = cm_rates(y_test, y_pred)
    net, v_tn, v_tp, c_fp, c_fn = dollars_from_rates(tn_r, fp_r, fn_r, tp_r)
    evs.append(net)
    details.append((th, net, cm, (tn_r, fp_r, fn_r, tp_r), (v_tn, v_tp, c_fp, c_fn)))

best_idx = int(np.argmax(evs))
best_th, best_net, best_cm, best_rates, best_parts = details[best_idx]
print(f"Best threshold by EV: {best_th:.2f}   Net annual value: ${best_net:,.0f}")
print("Rates (TN, FP, FN, TP):", [f"{r:.3f}" for r in best_rates])
print("Confusion matrix at best th:\n", best_cm)

# --- plot EV vs threshold ---
plt.figure(figsize=(7,4))
plt.plot(ths, evs, marker='o')
plt.axvline(best_th, ls='--', color='gray')
plt.title("Expected Annual Value vs Threshold")
plt.xlabel("Decision Probability Cutoff")
plt.ylabel("Projected Annual Net Profit ($ Millions)")
plt.tight_layout()
plt.show()

# --- business impact bars for the chosen threshold ---
labels = ["Savings from Avoiding Bad Sequels",
          "Profit from Successful Sequels",
          "Loss from Failed Sequels",
          "Missed Profit from Skipped \"Good\" Sequels"]
values = list(best_parts)
colors = ["#4CAF50", "#1E88E5", "#E53935", "#FB8C00"]

plt.figure(figsize=(9,5))
bars = plt.bar(labels, values, color=colors)
plt.axhline(0, color="black", lw=1)
plt.title(f"Business Impact (Per {DECISIONS_PER_YEAR} Decisions/Yr, th={best_th:.2f})")
plt.ylabel("Estimated $ Impact")
for b in bars:
    h = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, h + (0.02*max(values) if h>=0 else -0.02*max(values)),
             f"${h:,.0f}", ha="center", va="bottom" if h>=0 else "top", fontsize=10)
plt.text(0.5, 0.95, f"Projected Net Impact: ${best_net:,.0f} / year",
         transform=plt.gca().transAxes, ha="center",
         bbox=dict(facecolor="white", alpha=0.8))
plt.tight_layout()
plt.show()


# ## Don't Ignore - Final Impact Analysis

# Assumptions

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix

# ---------- Assumptions (edit these) ----------
VAL_TP = 35_000_000   # profit if we greenlight a truly good sequel
VAL_TN = 5_000_000    # savings if we correctly reject a bad sequel
COST_FP = 25_000_000  # loss if we greenlight a bad sequel
COST_FN = 35_000_000  # opp. cost if we miss a good sequel
DECISIONS_PER_YEAR = 15

# ---------- Helpers ----------
def dollars_from_rates(tn_rate, fp_rate, fn_rate, tp_rate,
                       n_decisions=DECISIONS_PER_YEAR,
                       val_tp=VAL_TP, val_tn=VAL_TN, cost_fp=COST_FP, cost_fn=COST_FN):
    TN = tn_rate * n_decisions
    FP = fp_rate * n_decisions
    FN = fn_rate * n_decisions
    TP = tp_rate * n_decisions
    value_tp = TP * val_tp
    value_tn = TN * val_tn
    cost_fp_ = FP * cost_fp
    cost_fn_ = FN * cost_fn
    net = value_tp + value_tn - cost_fp_ - cost_fn_
    # return parts in the order [TN, TP, FP, FN] as (+,+,-,-)
    return net, value_tn, value_tp, -cost_fp_, -cost_fn_

def cm_rates(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    return tn/total, fp/total, fn/total, tp/total, cm

def fmt_millions(x, pos):
    return f'${x/1e6:.0f}M'

# ---------- Sweep thresholds to maximize expected value ----------
ths = np.linspace(0.1, 0.9, 17)
evs, details = [], []

for th in ths:
    y_pred = (y_score >= th).astype(int)
    tn_r, fp_r, fn_r, tp_r, cm = cm_rates(y_test, y_pred)
    net, v_tn, v_tp, c_fp, c_fn = dollars_from_rates(tn_r, fp_r, fn_r, tp_r)
    evs.append(net)
    details.append((th, net, cm, (tn_r, fp_r, fn_r, tp_r), (v_tn, v_tp, c_fp, c_fn)))

best_idx = int(np.argmax(evs))
best_th, best_net, best_cm, best_rates, best_parts = details[best_idx]

print(f"Best threshold by EV: {best_th:.2f}   Net annual value: ${best_net:,.0f}")
print("Rates (TN, FP, FN, TP):", [f"{r:.3f}" for r in best_rates])
print("Confusion matrix at best th:\n", best_cm)

# Unpack parts for readability
val_TN, val_TP, cost_FP, cost_FN = best_parts
parts = [val_TN, val_TP, cost_FP, cost_FN]
labels_board = [
    "Savings from \nAvoiding Bad Sequels",
    "Profit from \nSuccessful Sequels",
    "Loss from \nFailed Sequels",
    "Missed Profit from \nSkipped Good Sequels"  # <-- fixed quotes
]
colors = ["#4CAF50", "#1E88E5", "#E53935", "#FB8C00"]


# #### Technical Version - Plots

# In[ ]:


# =================================================================
# TECHNICAL VERSION
# =================================================================

# 1) EV vs Threshold (technical)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(ths, evs, marker='o')
ax.axvline(best_th, ls='--', color='gray')
ax.set_title("Expected Annual Value vs Threshold")
ax.set_xlabel("Threshold")
ax.set_ylabel("Expected Annual Value ($)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
plt.tight_layout()
plt.show()

# 2) Business impact bars (technical) — FIXED
fig, ax = plt.subplots(figsize=(9,5))
bars = ax.bar(labels_board, parts, color=colors)
ax.axhline(0, color="black", lw=1)
ax.set_title(f"Business Impact (Per {DECISIONS_PER_YEAR} Decisions/Yr, th={best_th:.2f})")
ax.set_ylabel("Estimated $ Impact")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))

# --- wrap long x labels ---
ax.set_xticklabels(labels_board)  # set first so we can edit the Text objects
for lab in ax.get_xticklabels():
    lab.set_wrap(True)
    lab.set_ha("center")

# --- annotate bars, use symmetric padding based on absolute maxima ---
pad = 0.02 * max(map(abs, parts)) if parts else 0
for b in bars:
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width()/2,
        h + (pad if h >= 0 else -pad),
        f"${h:,.0f}",
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=10,
    )

ax.text(
    0.5, 0.95, f"Net: ${best_net:,.0f}/yr",
    transform=ax.transAxes, ha="center",
    bbox=dict(facecolor="white", alpha=0.8)
)

# give wrapped labels room
fig.subplots_adjust(bottom=0.20)
plt.tight_layout()
plt.show()


# #### Boardroom Version - Plots

# In[ ]:


# =================================================================
# BOARDROOM VERSION
# =================================================================

# 1) EV vs Threshold (boardroom polish)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(ths, evs, marker='o')
ax.axvline(best_th, ls='--', color='gray')
ax.set_title("Maximizing Annual Profit: Finding the Optimal Decision Cutoff")
ax.set_xlabel("Decision Probability Cutoff")
ax.set_ylabel("Projected Annual Net Profit ($ Millions)")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_millions))

# callout at peak
ax.annotate(f"Best Cutoff = {best_th:.2f}\n${best_net/1e6:.1f}M Net",
            xy=(best_th, best_net),
            xytext=(best_th+0.06, best_net*0.60),  # moved down
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
plt.tight_layout()
plt.show()

# 2) Business impact bars (boardroom polish) — FIXED
fig, ax = plt.subplots(figsize=(10,5))  # a touch taller for readability
fig.subplots_adjust(bottom=0.125)  # reserve space for footer
bars = ax.bar(labels_board, parts, color=colors)
ax.axhline(0, color="black", lw=1)
ax.set_title("Projected Financial Impact of Sequel Greenlighting "
             f"({DECISIONS_PER_YEAR} Decisions/Yr)")
ax.set_ylabel("Estimated $ Impact (Millions)")
ax.yaxis.set_major_formatter(FuncFormatter(fmt_millions))

# Percentage contribution labels
pct = [(v / best_net) if best_net != 0 else 0 for v in parts]
for b, v, p in zip(bars, parts, pct):
    h = b.get_height()
    txt = f"${abs(v)/1e6:.1f}M  ({p:+.0%} of net)"
    ax.text(b.get_x()+b.get_width()/2,
            h + (0.02*max(map(abs, parts)) if h>=0 else -0.02*max(map(abs, parts))),
            txt, ha="center", va="bottom" if h>=0 else "top", fontsize=10)

# Net impact text
ax.text(0.5, 0.95, f"Projected Net Impact: ${best_net/1e6:.1f}M / year",
        transform=ax.transAxes, ha="center",
        bbox=dict(facecolor="white", alpha=0.85))

# Footer in figure coordinates
footer_text = (f"Assumes: +${VAL_TP/1e6:.0f}M per profitable sequel, "
               f"+${VAL_TN/1e6:.0f}M saved per avoided bad sequel, "
               f"-${COST_FP/1e6:.0f}M per failed sequel, "
               f"-${COST_FN/1e6:.0f}M missed per skipped good sequel.")
fig.text(0.5, 0.02, footer_text, ha="center", va="center", fontsize=9)

plt.show()



# In[179]:


"""
BUSINESS VALUE CHECKER (based on EDA data)
- Time Saved (hours/month)
- Revenue/Profit Impact (from better sequel greenlighting + pricing optimization)
- Strategic Value proxies

Inputs you may want to tweak are in the ASSUMPTIONS section.
This script reads your EDA CSV and uses the observed dataset size and sequel prevalence
to ground the calculations.
"""

import math
import numpy as np
import pandas as pd

# =========================
# DATA INPUT
# =========================

catalog_size = len(df)
# if your target column is named differently, change here:
target_col = "is_sequel"
if target_col not in df.columns:
    raise ValueError(f"Expected '{target_col}' column in the dataset.")

# Observed sequel prevalence in your dataset (proxy for baseline precision if you greenlight naively)
sequel_prevalence = df[target_col].mean()  # e.g., ~0.164 as seen in the EDA
print(f"Catalog size detected: {catalog_size:,}")
print(f"Observed sequel prevalence (proxy baseline precision): {sequel_prevalence:.3f}")

# =========================
# ASSUMPTIONS (EDIT THESE)
# =========================
# Analyst time assumptions (per month)
analysts = 2                       # analysts who did market/competitive analysis manually
hours_per_week_manual = 8          # hours per analyst per week before automation
automation_coverage = 0.75         # fraction of the work automated (0..1)
weeks_per_month = 4.345            # avg weeks/month
hourly_cost_fully_loaded = 55      # $/hr (wages + overhead) — optional for $ conversion

# Sequel greenlighting assumptions
annual_catalog_size = 2000         # titles evaluated yearly
sequel_candidates_share = 0.15     # share of titles that become sequel candidates
greenlights_made_rate = 0.30       # fraction of candidates that get greenlit (decision volume)
# Precision improvement: use your model precision (class 1) from EDA vs baseline
# Baseline precision ≈ sequel_prevalence (naive/no-model). Improved precision from your model (e.g., 0.55 from EDA report)
baseline_precision = float(sequel_prevalence)
improved_precision = 0.55          # <- put your RandomForest class-1 precision from EDA here

# Profit per successful greenlight (incremental contribution over status quo)
avg_profit_per_success = 120_000   # $

# Pricing optimization assumptions
share_of_titles_repriced = 0.25    # fraction of titles affected by pricing optimization
avg_annual_profit_per_title = 20_000  # baseline annual profit per title ($)
avg_price_uplift_pct = 0.03        # % improvement in profit from optimized pricing (e.g., 3%)

# Deployment/ops costs for ROI
one_off_build_cost = 15_000
monthly_run_cost = 400
months_considered = 12

# Strategic value proxies (for reporting)
time_to_insight_reduction_pct = 0.70   # e.g., 70% faster with dashboard
decision_coverage_pct = 0.95           # % of decisions now backed by model outputs
accessibility_note = "Global web access via Hugging Face (role-gated, no VPN)"

# =========================
# CALCULATIONS
# =========================
# 1) TIME SAVED (hours/month)
monthly_hours_manual = analysts * hours_per_week_manual * weeks_per_month
monthly_hours_saved = monthly_hours_manual * automation_coverage
monthly_cost_saved = monthly_hours_saved * hourly_cost_fully_loaded

# 2) REVENUE/PROFIT IMPACT
annual_sequel_candidates = annual_catalog_size * sequel_candidates_share
greenlights_made = annual_sequel_candidates * greenlights_made_rate

# Incremental correct greenlights due to higher precision (holding volume constant)
precision_uplift = max(improved_precision - baseline_precision, 0)
incremental_successes = greenlights_made * precision_uplift
annual_profit_from_better_greenlighting = incremental_successes * avg_profit_per_success

# Pricing optimization
titles_repriced = annual_catalog_size * share_of_titles_repriced
annual_profit_from_pricing = titles_repriced * avg_annual_profit_per_title * avg_price_uplift_pct

annual_total_profit_uplift = annual_profit_from_better_greenlighting + annual_profit_from_pricing

# 3) STRATEGIC VALUE (proxies only)
strategic_metrics = {
    "time_to_insight_reduction_pct": time_to_insight_reduction_pct,
    "decision_coverage_pct": decision_coverage_pct,
    "accessibility": accessibility_note
}

# 4) ROI over horizon
total_run_cost = monthly_run_cost * months_considered
total_time_savings_value = monthly_cost_saved * months_considered
profit_horizon = annual_total_profit_uplift * (months_considered / 12)

benefits_total = total_time_savings_value + profit_horizon
costs_total = one_off_build_cost + total_run_cost
net_benefit = benefits_total - costs_total
roi_x = (net_benefit / costs_total) if costs_total > 0 else math.inf

# =========================
# OUTPUT
# =========================
print("\n=== TIME SAVED ===")
print(f"Monthly hours saved: {monthly_hours_saved:,.1f} hrs")
print(f"Monthly cost saved:  ${monthly_cost_saved:,.0f}")

print("\n=== PROFIT IMPACT (Annualized) ===")
print(f"Sequel candidates/year:                 {annual_sequel_candidates:,.0f}")
print(f"Greenlights made/year (volume):         {greenlights_made:,.0f}")
print(f"Baseline precision (from EDA prevalence): {baseline_precision:.2f}")
print(f"Improved precision (model):             {improved_precision:.2f}")
print(f"Incremental successful greenlights:     {incremental_successes:,.1f}")
print(f"Profit from better greenlighting:       ${annual_profit_from_better_greenlighting:,.0f}")
print(f"Profit from pricing optimization:       ${annual_profit_from_pricing:,.0f}")
print(f"Total annual profit uplift:             ${annual_total_profit_uplift:,.0f}")

print("\n=== STRATEGIC VALUE (Proxies) ===")
print(f"Time-to-insight reduction:  {strategic_metrics['time_to_insight_reduction_pct']:.0%}")
print(f"Decision coverage:          {strategic_metrics['decision_coverage_pct']:.0%}")
print(f"Accessibility:              {strategic_metrics['accessibility']}")

print(f"\n=== ROI over {months_considered} months ===")
print(f"Total benefits:  ${benefits_total:,.0f}")
print(f"Total costs:     ${costs_total:,.0f}")
print(f"Net benefit:     ${net_benefit:,.0f}")
print(f"ROI:             {roi_x:.1f}x")

# =========================
# OPTIONAL: QUICK SENSITIVITY TABLE
# =========================
sens = []
for uplift in [0.02, 0.05, 0.10, 0.15]:
    for imp_prec in [0.40, 0.50, 0.55, 0.60]:  # try different model precisions
        prec_uplift = max(imp_prec - baseline_precision, 0)
        incr_succ = greenlights_made * prec_uplift
        profit_gl = incr_succ * avg_profit_per_success
        profit_pr = titles_repriced * avg_annual_profit_per_title * uplift
        annual_profit = profit_gl + profit_pr
        profit_h = annual_profit * (months_considered / 12)
        benefits = total_time_savings_value + profit_h
        roi_val = (benefits - costs_total) / costs_total
        sens.append({
            "Price uplift %": f"{int(uplift*100)}%",
            "Model precision": f"{imp_prec:.2f}",
            "Annual profit uplift ($)": round(annual_profit),
            f"ROI {months_considered} mo (x)": round(roi_val, 2)
        })

sens_df = pd.DataFrame(sens)
print("\n=== SENSITIVITY: ROI vs Model Precision & Price Uplift ===")
print(sens_df.to_string(index=False))


# In[181]:


# === BUSINESS VALUE PLOTS (Matplotlib only) ===
# Generates four separate figures to support your CTO slide:
# 1) Time Saved (bar)
# 2) Annual Profit Uplift Breakdown (stacked bar)
# 3) Cumulative Net Benefit Over Time (line)
# 4) Sensitivity Heatmap (ROI vs Model Precision & Price Uplift)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# DATA INPUT
# =========================

catalog_size = len(df)
target_col = "is_sequel"
if target_col not in df.columns:
    raise ValueError(f"Expected '{target_col}' column in the dataset.")

sequel_prevalence = df[target_col].mean()  # ~0.164 per your EDA
print(f"Catalog size detected: {catalog_size:,}")
print(f"Observed sequel prevalence: {sequel_prevalence:.3f}")

# =========================
# ASSUMPTIONS (EDIT THESE)
# =========================

# Analyst time assumptions (per month)
analysts = 2                       # analysts who did market/competitive analysis manually
hours_per_week_manual = 8          # hours per analyst per week before automation
automation_coverage = 0.75         # fraction of the work automated (0..1)
weeks_per_month = 4.345            # avg weeks/month
hourly_cost_fully_loaded = 55      # $/hr (optional for $ conversion)

# Sequel greenlighting assumptions
annual_catalog_size = 2000         # titles evaluated yearly
sequel_candidates_share = 0.15     # share that become sequel candidates
greenlights_made_rate = 0.30       # fraction of candidates that get greenlit
baseline_precision = float(sequel_prevalence)  # naive precision
improved_precision = 0.55          # from your EDA classification report (class-1 precision)

avg_profit_per_success = 120_000   # $ profit per successful greenlight

# Pricing optimization assumptions
share_of_titles_repriced = 0.25
avg_annual_profit_per_title = 20_000
avg_price_uplift_pct = 0.03        # 3% lift

# Deployment/ops costs for ROI
one_off_build_cost = 15_000
monthly_run_cost = 400
months_considered = 12

# =========================
# CALCULATIONS
# =========================

# 1) TIME SAVED
monthly_hours_manual = analysts * hours_per_week_manual * weeks_per_month
monthly_hours_saved = monthly_hours_manual * automation_coverage
monthly_cost_saved = monthly_hours_saved * hourly_cost_fully_loaded

automated_hours = monthly_hours_manual - monthly_hours_saved

# 2) PROFIT IMPACT
annual_sequel_candidates = annual_catalog_size * sequel_candidates_share
greenlights_made = annual_sequel_candidates * greenlights_made_rate

precision_uplift = max(improved_precision - baseline_precision, 0)
incremental_successes = greenlights_made * precision_uplift
annual_profit_from_better_greenlighting = incremental_successes * avg_profit_per_success

titles_repriced = annual_catalog_size * share_of_titles_repriced
annual_profit_from_pricing = titles_repriced * avg_annual_profit_per_title * avg_price_uplift_pct

annual_total_profit_uplift = annual_profit_from_better_greenlighting + annual_profit_from_pricing

# 3) ROI trajectory
months = np.arange(1, months_considered + 1)
monthly_profit_uplift = annual_total_profit_uplift / 12.0
monthly_time_savings_value = monthly_cost_saved
monthly_benefit = monthly_profit_uplift + monthly_time_savings_value
cumulative_benefit = monthly_benefit * months
cumulative_costs = one_off_build_cost + monthly_run_cost * months
cumulative_net_benefit = cumulative_benefit - cumulative_costs

# =========================
# PLOTS
# =========================

# --- Plot 1: Time Saved (Manual vs Automated) ---
plt.figure(figsize=(8, 5))
x_labels = ["Manual Process", "Automated Process"]
y_vals = [monthly_hours_manual, automated_hours]
bars = plt.bar(x_labels, y_vals)
plt.title("Monthly Hours for Market & Competitive Analysis")
plt.ylabel("Hours per Month")
plt.ylim(0, max(y_vals) * 1.2)
for i, v in enumerate(y_vals):
    plt.text(i, v + max(y_vals) * 0.05, f"{v:,.1f} hrs", ha="center")
reduction_pct = (1 - automated_hours / monthly_hours_manual) if monthly_hours_manual > 0 else 0
plt.text(0.5, max(y_vals) * 1.05, f"Reduction: {reduction_pct:.0%}", ha="center")
plt.tight_layout()
# plt.savefig("plot_time_saved.png", dpi=200)

# --- Plot 2: Annual Profit Uplift Breakdown (stacked bar) ---
plt.figure(figsize=(8, 5))
base = annual_profit_from_better_greenlighting
plt.bar(["Annual Profit Uplift"], [base], label="Better Greenlighting")
plt.bar(["Annual Profit Uplift"], [annual_profit_from_pricing], bottom=[base], label="Optimized Pricing")
plt.title("Annual Profit Uplift Breakdown")
plt.ylabel("Profit ($)")
plt.legend()
total_label = base + annual_profit_from_pricing
plt.text(0, total_label * 1.02, f"Total: ${total_label:,.0f}", ha="center")
plt.tight_layout()
# plt.savefig("plot_profit_uplift.png", dpi=200)

# --- Plot 3: Cumulative Net Benefit Over Time (ROI trajectory) ---
plt.figure(figsize=(9, 5))
plt.plot(months, cumulative_net_benefit, marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title(f"Cumulative Net Benefit Over Time (0–{months_considered} months)")
plt.xlabel("Months")
plt.ylabel("Net Benefit ($)")
# Payback month annotation
payback_idx = np.argmax(cumulative_net_benefit > 0)
if cumulative_net_benefit[0] > 0:
    payback_month = 1
elif cumulative_net_benefit[-1] <= 0:
    payback_month = None
else:
    payback_month = payback_idx + 1 if payback_idx > 0 else None
if payback_month:
    y_val = cumulative_net_benefit[payback_month - 1]
    plt.scatter([payback_month], [y_val], s=60)
    plt.text(payback_month, y_val, f" Payback ~Month {payback_month}", va="bottom")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.savefig("plot_roi_over_time.png", dpi=200)

# --- Plot 4: Sensitivity Heatmap (ROI vs Model Precision & Price Uplift) ---
precisions = np.array([0.40, 0.50, 0.55, 0.60])
uplifts = np.array([0.02, 0.05, 0.10, 0.15])
roi_matrix = np.zeros((len(uplifts), len(precisions)))

for i, uplift in enumerate(uplifts):
    for j, imp_prec in enumerate(precisions):
        prec_uplift = max(imp_prec - baseline_precision, 0)
        incr_succ = greenlights_made * prec_uplift
        profit_gl = incr_succ * avg_profit_per_success
        profit_pr = titles_repriced * avg_annual_profit_per_title * uplift
        annual_profit = profit_gl + profit_pr
        profit_h = annual_profit * (months_considered / 12)
        benefits = monthly_cost_saved * months_considered + profit_h
        costs = one_off_build_cost + monthly_run_cost * months_considered
        roi_val = (benefits - costs) / costs
        roi_matrix[i, j] = roi_val

plt.figure(figsize=(7, 5))
im = plt.imshow(roi_matrix, aspect="auto")
plt.colorbar(im, label=f"ROI over {months_considered} months (x)")
plt.xticks(range(len(precisions)), [f"{p:.2f}" for p in precisions])
plt.yticks(range(len(uplifts)), [f"{int(u*100)}%" for u in uplifts])
plt.xlabel("Model Precision (Class 1)")
plt.ylabel("Price Uplift (%)")
plt.title("ROI Sensitivity: Precision vs Price Uplift")
# Annotate cells
for i in range(len(uplifts)):
    for j in range(len(precisions)):
        plt.text(j, i, f"{roi_matrix[i, j]:.2f}", ha="center", va="center")
plt.tight_layout()
# plt.savefig("plot_roi_sensitivity.png", dpi=200)

plt.show()

# ---- Console summary (optional) ----
print("\n--- SUMMARY ---")
print(f"Monthly hours saved: {monthly_hours_saved:,.1f} hrs  (~{reduction_pct:.0%} reduction)")
print(f"Annual profit uplift: ${annual_total_profit_uplift:,.0f} "
      f"= GL ${annual_profit_from_better_greenlighting:,.0f} + PR ${annual_profit_from_pricing:,.0f}")

