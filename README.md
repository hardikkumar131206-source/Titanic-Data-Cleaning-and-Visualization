#  Titanic Data Cleaning & Visualization

A complete data science pipeline on the classic Titanic dataset — covering raw data ingestion, cleaning, feature engineering, outlier handling, and multi-chart visual storytelling using Python.

---

##  Project Overview

This project takes the raw Titanic passenger dataset (891 rows, 12 columns) and transforms it into clean, analysis-ready data while generating 5 publication-quality charts that reveal the key survival patterns aboard the RMS Titanic.

| Stage | What happens |
|---|---|
| **Ingest** | Download raw CSV from public source |
| **Audit** | Identify missing values, duplicates, outliers |
| **Clean** | Impute, encode, and cap values |
| **Engineer** | Create 5 new features from existing columns |
| **Visualize** | Generate 5 themed chart sets as PNG files |
| **Export** | Save cleaned CSV for downstream use |

---

##  Project Structure

```
titanic-analysis/
│
├── titanic_analysis.py         # Main pipeline script
├── titanic_cleaned.csv         # Output: cleaned dataset
│
├── charts/
│   ├── chart1_overview_dashboard.png   # 6-panel survival overview
│   ├── chart2_survival_rates.png       # Survival % by gender, class, age group
│   ├── chart3_heatmaps.png             # Correlation matrix + class×gender heatmap
│   ├── chart4_family_fare_cabin.png    # Family size, fare band, cabin insights
│   └── chart5_data_quality.png        # Before/after missing value report
│
└── README.md
```

---

##  Data Cleaning Steps

### Missing Values

| Column | Missing | Strategy |
|---|---|---|
| `Age` | 177 (19.9%) | Filled with **median grouped by Pclass × Sex** |
| `Cabin` | 687 (77.1%) | Converted to binary `HasCabin` flag — too sparse to impute |
| `Embarked` | 2 (0.2%) | Filled with **mode** (Southampton) |

### Duplicates
Zero duplicate rows detected in the raw data.

### Outliers
`Fare` had **116 outliers** detected via IQR method. Values were **kept** in the dataset but **capped at the 97th percentile** for plotting to avoid distorted chart scales.

---

##  Feature Engineering

Five new features were derived from existing columns:

| Feature | Description |
|---|---|
| `HasCabin` | 1 if cabin info exists, 0 otherwise |
| `FamilySize` | `SibSp + Parch + 1` (includes self) |
| `IsAlone` | 1 if travelling solo (FamilySize == 1) |
| `AgeGroup` | Binned: Child / Teen / Adult / Middle-Age / Senior |
| `FareBand` | Quartile buckets: Low / Mid / High / Very High |

---

##  Charts Generated

### Chart 1 — Overview Dashboard
Six-panel grid covering survival counts, survival by gender, by passenger class, age distribution, fare distribution, and port of embarkation.



---

### Chart 2 — Survival Rates (%)
Bar charts showing survival rate percentage across gender, class, and age group with a 50% reference line.



---

### Chart 3 — Heatmaps
- **Correlation matrix** across all numeric features
- **Pivot heatmap** of survival rate % broken down by Pclass × Gender


---

### Chart 4 — Family, Fare & Cabin Deep Dive
- Survival rate by family size (line plot)
- Survival rate by fare band (bar chart)
- Survival count by cabin record availability



---

### Chart 5 — Data Quality Report
Side-by-side missing value comparison — before and after cleaning — with percentage bars per column.



---

##  Key Findings

| Insight | Value |
|---|---|
| Overall survival rate | **38.4%** |
| Female survival rate | **74.2%** |
| Male survival rate | **18.9%** |
| 1st class survival rate | **63.0%** |
| 3rd class survival rate | **24.2%** |
| Avg fare — survived | **£48.40** |
| Avg fare — died | **£22.12** |
| Best family size for survival | **2–4 members** |

> **Bottom line:** Gender and passenger class were the two strongest predictors of survival. Women and first-class passengers had dramatically better odds — consistent with the "women and children first" evacuation protocol and the physical accessibility of lifeboats by deck.

---

##  Tech Stack

- **Python 3.x**
- [pandas](https://pandas.pydata.org/) — data loading, cleaning, feature engineering
- [matplotlib](https://matplotlib.org/) — chart rendering and layout
- [seaborn](https://seaborn.pydata.org/) — heatmaps and styled plots
- [numpy](https://numpy.org/) — numerical operations

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/titanic-analysis.git
cd titanic-analysis
```

### 2. Install dependencies
```bash
pip install pandas matplotlib seaborn numpy
```

### 3. Run the pipeline
```bash
python titanic_analysis.py
```

Charts will be saved as `.png` files and the cleaned dataset as `titanic_cleaned.csv` in the working directory.

---

##  Dataset

Raw data sourced from the [DataScienceDojo Datasets repository](https://github.com/datasciencedojo/datasets) — a widely used version of the original Titanic passenger manifest.

- **Rows:** 891 passengers
- **Columns:** 12 (raw) → 16 (after feature engineering)
- **Target variable:** `Survived` (0 = No, 1 = Yes)

---

##  Author

**Hardik Kumar**
📧 hardikumar131206@gmail.com
🔗 [linkedin.com/in](https://www.linkedin.com/in/hardik-kumar-7631a832b)
🐙 [github.com](https://github.com/hardikkumar131206-source)

---

