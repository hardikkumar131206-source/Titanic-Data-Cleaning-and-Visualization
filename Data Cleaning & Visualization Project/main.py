"""
Titanic Dataset - Data Cleaning & Visualization
================================================
Raw data from: https://github.com/datasciencedojo/datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# CONFIG

PALETTE = {
    "navy":    "#1B2A4A",
    "steel":   "#3A5A8C",
    "ice":     "#A8C4D8",
    "rust":    "#C0392B",
    "gold":    "#E8B84B",
    "light":   "#F5F7FA",
    "survived": "#3A8C5A",
    "died":     "#C0392B",
}
sns.set_theme(style="white", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": PALETTE["light"],
    "axes.facecolor":   PALETTE["light"],
    "axes.edgecolor":   PALETTE["navy"],
    "axes.labelcolor":  PALETTE["navy"],
    "xtick.color":      PALETTE["navy"],
    "ytick.color":      PALETTE["navy"],
    "text.color":       PALETTE["navy"],
    "font.size":        11,
})


# 1. LOAD RAW DATA

print("=" * 55)
print("  TITANIC DATA CLEANING & VISUALIZATION PIPELINE")
print("=" * 55)

df_raw = pd.read_csv("titanic_cleaned.csv")
print(f"\n[LOAD]  Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")


# 2. CLEANING REPORT

print("\n[CLEAN] Missing values before:")
missing_before = df_raw.isnull().sum()
print(missing_before[missing_before > 0].to_string())

duplicates = df_raw.duplicated().sum()
print(f"\n[CLEAN] Duplicate rows: {duplicates}")

df = df_raw.copy()

# 2a. Age — fill with median grouped by Pclass & Sex
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# 2b. Cabin handling
if "Cabin" in df.columns:
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df.drop(columns=["Cabin"], inplace=True)
else:
    print("[INFO] Cabin column not found — skipping cabin processing.")

# 2c. Embarked — fill 2 missing with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 2d. Remove duplicates
df.drop_duplicates(inplace=True)

# 2e. Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"]   = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle-Age", "Senior"]
)
df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Mid", "High", "Very High"])

print(f"\n[CLEAN] Missing values after: {df.isnull().sum().sum()} total remaining")
print(f"[CLEAN] Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[CLEAN] New features: HasCabin, FamilySize, IsAlone, AgeGroup, FareBand")


# 3. OUTLIER DETECTION (Fare)

Q1, Q3 = df["Fare"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df["Fare"] < Q1 - 1.5 * IQR) | (df["Fare"] > Q3 + 1.5 * IQR)]
print(f"\n[OUTLIERS] Fare outliers detected: {len(outliers)} rows (kept, capped for plots)")
df["FareCapped"] = df["Fare"].clip(upper=df["Fare"].quantile(0.97))


# HELPER

surv_colors = [PALETTE["died"], PALETTE["survived"]]

def add_bar_labels(ax, fmt="{:.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9,
                        color=PALETTE["navy"], fontweight="bold")

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=PALETTE["navy"], pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


# CHART 1 — OVERVIEW DASHBOARD

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Titanic Survival — Overview Dashboard",
             fontsize=18, fontweight="bold", color=PALETTE["navy"], y=1.01)

# 1a Survival count
ax = axes[0, 0]
counts = df["Survived"].value_counts().sort_index()
bars = ax.bar(["Did Not Survive", "Survived"], counts.values,
              color=[PALETTE["died"], PALETTE["survived"]], width=0.5, edgecolor="white", linewidth=1.5)
add_bar_labels(ax)
style_ax(ax, "Overall Survival Count", ylabel="Passengers")

# 1b Survival by Sex
ax = axes[0, 1]
sex_surv = df.groupby(["Sex", "Survived"]).size().unstack()
sex_surv.plot(kind="bar", ax=ax, color=surv_colors, edgecolor="white",
              linewidth=1.2, width=0.6)
ax.set_xticklabels(["Female", "Male"], rotation=0)
add_bar_labels(ax)
style_ax(ax, "Survival by Gender", ylabel="Passengers")
ax.legend(["Died", "Survived"], frameon=False)

# 1c Survival by Pclass
ax = axes[0, 2]
cls_surv = df.groupby(["Pclass", "Survived"]).size().unstack()
cls_surv.plot(kind="bar", ax=ax, color=surv_colors, edgecolor="white",
              linewidth=1.2, width=0.6)
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"], rotation=0)
add_bar_labels(ax)
style_ax(ax, "Survival by Passenger Class", ylabel="Passengers")
ax.legend(["Died", "Survived"], frameon=False)

# 1d Age distribution
ax = axes[1, 0]
for s, color, label in [(0, PALETTE["died"], "Died"), (1, PALETTE["survived"], "Survived")]:
    ax.hist(df[df["Survived"] == s]["Age"].dropna(), bins=25,
            alpha=0.6, color=color, label=label, edgecolor="white")
style_ax(ax, "Age Distribution by Survival", xlabel="Age", ylabel="Count")
ax.legend(frameon=False)

# 1e Fare distribution (capped)
ax = axes[1, 1]
for s, color, label in [(0, PALETTE["died"], "Died"), (1, PALETTE["survived"], "Survived")]:
    ax.hist(df[df["Survived"] == s]["FareCapped"], bins=25,
            alpha=0.6, color=color, label=label, edgecolor="white")
style_ax(ax, "Fare Distribution by Survival\n(97th pct capped)", xlabel="Fare (£)", ylabel="Count")
ax.legend(frameon=False)

# 1f Embarkation port
ax = axes[1, 2]
port_surv = df.groupby(["Embarked", "Survived"]).size().unstack()
port_surv.plot(kind="bar", ax=ax, color=surv_colors, edgecolor="white",
               linewidth=1.2, width=0.6)
ax.set_xticklabels(["Cherbourg", "Queenstown", "Southampton"], rotation=0)
add_bar_labels(ax)
style_ax(ax, "Survival by Port of Embarkation", ylabel="Passengers")
ax.legend(["Died", "Survived"], frameon=False)

plt.tight_layout()
plt.savefig("chart1_overview_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
plt.close()
print("\n[CHART 1] Saved → chart1_overview_dashboard.png")


# CHART 2 — SURVIVAL RATES (%)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Titanic — Survival Rates (%)",
             fontsize=16, fontweight="bold", color=PALETTE["navy"])

def rate_bar(ax, group_col, title, labels=None):
    rates = df.groupby(group_col)["Survived"].mean() * 100
    if labels:
        rates.index = labels
    bars = ax.bar(rates.index, rates.values,
                  color=[PALETTE["steel"] if v < 50 else PALETTE["survived"] for v in rates.values],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for p, v in zip(bars, rates.values):
        ax.text(p.get_x() + p.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=PALETTE["navy"])
    ax.axhline(50, color=PALETTE["rust"], linewidth=1.2,
               linestyle="--", alpha=0.7, label="50% line")
    ax.set_ylim(0, 100)
    style_ax(ax, title, ylabel="Survival Rate (%)")

rate_bar(axes[0], "Sex",    "By Gender",         ["Female", "Male"])
rate_bar(axes[1], "Pclass", "By Passenger Class",["1st", "2nd", "3rd"])
rate_bar(axes[2], "AgeGroup","By Age Group")
axes[2].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("chart2_survival_rates.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
plt.close()
print("[CHART 2] Saved → chart2_survival_rates.png")


# CHART 3 — HEATMAP & CORRELATION

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Titanic — Correlation & Class × Gender Heatmap",
             fontsize=15, fontweight="bold", color=PALETTE["navy"])

# 3a correlation heatmap
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch",
            "Fare", "FamilySize", "HasCabin", "IsAlone"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[0],
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"size": 8}, center=0,
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Feature Correlation Matrix",
                  fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=10)
axes[0].tick_params(axis="x", rotation=45)

# 3b pivot heatmap: survival rate by Class × Sex
pivot = df.pivot_table("Survived", index="Pclass", columns="Sex", aggfunc="mean") * 100
pivot.index = ["1st Class", "2nd Class", "3rd Class"]
pivot.columns = ["Female", "Male"]
sns.heatmap(pivot, ax=axes[1], annot=True, fmt=".1f",
            cmap="RdYlGn", linewidths=2,
            annot_kws={"size": 14, "fontweight": "bold"},
            cbar_kws={"label": "Survival Rate (%)", "shrink": 0.8},
            vmin=0, vmax=100)
axes[1].set_title("Survival Rate % by Class & Gender",
                  fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=10)
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("chart3_heatmaps.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
plt.close()
print("[CHART 3] Saved → chart3_heatmaps.png")


# CHART 4 — FAMILY, FARE & CABIN DEEP DIVE

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Titanic — Family, Fare Band & Cabin Insights",
             fontsize=15, fontweight="bold", color=PALETTE["navy"])

# 4a family size survival rate
ax = axes[0]
fam_rate = df.groupby("FamilySize")["Survived"].mean() * 100
ax.plot(fam_rate.index, fam_rate.values,
        marker="o", color=PALETTE["steel"], linewidth=2.5, markersize=8,
        markerfacecolor=PALETTE["gold"], markeredgecolor=PALETTE["navy"])
for x, y in zip(fam_rate.index, fam_rate.values):
    ax.text(x, y + 2, f"{y:.0f}%", ha="center", fontsize=8.5,
            color=PALETTE["navy"], fontweight="bold")
ax.axhline(50, color=PALETTE["rust"], linewidth=1, linestyle="--", alpha=0.6)
ax.set_xticks(fam_rate.index)
style_ax(ax, "Survival Rate by Family Size",
         xlabel="Family Size (self + relatives)", ylabel="Survival Rate (%)")
ax.set_ylim(0, 100)

# 4b fare band survival rate
ax = axes[1]
fare_surv = df.groupby("FareBand", observed=True)["Survived"].mean() * 100
bars = ax.bar(fare_surv.index, fare_surv.values,
              color=[PALETTE["rust"], PALETTE["ice"], PALETTE["steel"], PALETTE["survived"]],
              edgecolor="white", linewidth=1.5, width=0.6)
for p, v in zip(bars, fare_surv.values):
    ax.text(p.get_x() + p.get_width() / 2, v + 1,
            f"{v:.1f}%", ha="center", fontsize=10,
            fontweight="bold", color=PALETTE["navy"])
ax.axhline(50, color=PALETTE["rust"], linewidth=1, linestyle="--", alpha=0.6)
ax.set_ylim(0, 100)
style_ax(ax, "Survival Rate by Fare Band", ylabel="Survival Rate (%)")

# 4c cabin vs no cabin
ax = axes[2]
cabin_data = df.groupby(["HasCabin", "Survived"]).size().unstack()
cabin_data.index = ["No Cabin Info", "Has Cabin Info"]
cabin_data.plot(kind="bar", ax=ax, color=surv_colors,
                edgecolor="white", linewidth=1.2, width=0.5)
ax.set_xticklabels(cabin_data.index, rotation=0)
add_bar_labels(ax)
style_ax(ax, "Survival by Cabin Record", ylabel="Passengers")
ax.legend(["Died", "Survived"], frameon=False)

plt.tight_layout()
plt.savefig("chart4_family_fare_cabin.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
plt.close()
print("[CHART 4] Saved → chart4_family_fare_cabin.png")


# CHART 5 — MISSING VALUE REPORT

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(PALETTE["light"])
fig.suptitle("Titanic — Data Quality Report",
             fontsize=15, fontweight="bold", color=PALETTE["navy"])

# 5a before cleaning
ax = axes[0]
miss_before = df_raw.isnull().mean() * 100
miss_before = miss_before[miss_before > 0].sort_values(ascending=True)
bars = ax.barh(miss_before.index, miss_before.values,
               color=PALETTE["rust"], edgecolor="white", linewidth=1.2)
for p in bars:
    w = p.get_width()
    ax.text(w + 0.5, p.get_y() + p.get_height() / 2,
            f"{w:.1f}%", va="center", fontsize=9,
            color=PALETTE["navy"], fontweight="bold")
ax.set_xlim(0, 100)
style_ax(ax, "Missing Values — BEFORE Cleaning (%)", xlabel="Missing %")
ax.spines[["top", "right"]].set_visible(False)

# 5b after cleaning
ax = axes[1]
miss_after = df.isnull().mean() * 100
miss_after_nonzero = miss_after[miss_after > 0]
if miss_after_nonzero.empty:
    ax.text(0.5, 0.5, "✓  No Missing Values\nAfter Cleaning!",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=PALETTE["survived"],
            transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
else:
    bars = ax.barh(miss_after_nonzero.index, miss_after_nonzero.values,
                   color=PALETTE["gold"], edgecolor="white")
    ax.set_xlim(0, 100)
ax.set_title("Missing Values — AFTER Cleaning (%)",
             fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=10)

plt.tight_layout()
plt.savefig("chart5_data_quality.png",
            dpi=150, bbox_inches="tight", facecolor=PALETTE["light"])
plt.close()
print("[CHART 5] Saved → chart5_data_quality.png")


# SAVE CLEANED CSV

df.to_csv("titanic_cleaned_output.csv", index=False)
print("\n[CSV]   Saved → titanic_cleaned.csv")


# SUMMARY STATS

print("\n" + "=" * 55)
print("  KEY FINDINGS")
print("=" * 55)
surv_rate = df["Survived"].mean() * 100
print(f"  Overall survival rate     : {surv_rate:.1f}%")
print(f"  Female survival rate      : {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")
print(f"  Male survival rate        : {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"  1st class survival rate   : {df[df['Pclass']==1]['Survived'].mean()*100:.1f}%")
print(f"  3rd class survival rate   : {df[df['Pclass']==3]['Survived'].mean()*100:.1f}%")
print(f"  Avg age (survived)        : {df[df['Survived']==1]['Age'].mean():.1f} yrs")
print(f"  Avg age (died)            : {df[df['Survived']==0]['Age'].mean():.1f} yrs")
print(f"  Avg fare (survived)       : £{df[df['Survived']==1]['Fare'].mean():.2f}")
print(f"  Avg fare (died)           : £{df[df['Survived']==0]['Fare'].mean():.2f}")
print("=" * 55)
print("  All charts saved to /mnt/user-data/outputs/")
print("=" * 55)