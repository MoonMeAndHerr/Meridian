# 🌐 MERIDIAN: Strategic Underemployment & Vulnerability Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://meridiansru.streamlit.app)
[![Competition](https://img.shields.io/badge/Competition-DAX_Challenge_2026-blue.svg)](https://dax.umpsa.edu.my/)
[![Track](https://img.shields.io/badge/Domain-Economy_(DOSM)-teal.svg)]()
[![Target](https://img.shields.io/badge/Focus-Negeri_Pahang-gold.svg)]()
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
Standard economic indicators often paint a misleading picture of regional prosperity. In **Pahang, Malaysia**, official unemployment rates remain low (~1.9%), yet households face mounting financial distress. 

**MERIDIAN** is an advanced policy-intelligence and analytical dashboard built to uncover the hidden **"Working Poor Trap."** By cross-referencing official Department of Statistics Malaysia (DOSM) datasets, MERIDIAN exposes the structural disconnect between job creation, skill-level supply, and skyrocketing living costs—providing state policymakers with a data-driven lens for early intervention and targeted resource allocation.

---

## 🏗️ Dashboard Architecture & Key Metrics

The application is structured into a logical, public-friendly storytelling framework:

### 1. Row 1: The Macro Contradiction (Employment vs. Lived Reality)
*   **Average Unemployment Rate (`LMR Sheet A.7.2`):** Tracks headline joblessness, demonstrating a healthy downward trend.
*   **Incidence of Absolute Poverty, $H(\%)$ (`MPI Jadual 1`):** Highlights multi-year household vulnerability trajectories through interpolated timelines.
*   **Skill-Related Underemployment (SRU):** Tracks the percentage of tertiary graduates forced into low-skill positions due to structural market gaps.

### 2. Row 2: The Economic Squeeze (Cost of Living vs. Hardship)
*   **Cost of Living (PAKW Indicators):** District-level breakdown across all 11 administrative districts of Pahang, filterable by household size (1 to 4 persons, plus state aggregate).
*   **Multidimensional Poverty Drivers & Hardships (`MPI Jadual 3`):** Explicitly separates structural root causes (Income Deprivation) from daily lived consequences (Water access, crowdedness, sanitation).

### 3. Row 3: The Root Cause (Sectoral Skill-Level Mismatch)
*   **Labor Market Demand (`LMR Sheets B.4, B.5, B.6`):** Compares Positions Filled, Current Vacancies, and New Jobs Created across all **18 DOSM economic sectors** and broken down by **Skill Level (Skilled, Semi-Skilled, Low-Skilled)**. Mathematically proves the scarcity of high-value roles for educated talent.

---

## 🤖 Integrated AI & Policy Modules

MERIDIAN features a modular operations panel allowing stakeholders to simulate future economic scenarios:
*   **Phase 1: Predictive Analytics Module:** Uses time-series regression algorithms (comparing baseline Linear Regression vs. non-linear XGBoost) to forecast the 2030 SRU trajectory and deploys **K-Means Clustering ($K=3$)** to automatically segment Pahang's districts into a *Traffic Light Risk Matrix* (Stable, Squeezed, and Critical Red Zones).
*   **Phase 2: Prescriptive Policy Simulator:** An interactive budget optimization engine enabling state planners to simulate grant allocations and immediately calculate projected reductions in underemployment.

---

## 📊 Dataset Sources & Integrity
All insights are synthesized directly from official Malaysian economic publications:
1.  **Labour Market Review (LMR), Q1 2026** (DOSM)
2.  **Multidimensional Poverty Index (MPI) Report** (DOSM)
3.  **Cost of Living Indicators (PAKW), 2024** (DOSM)
4.  **Skill-Related Underemployment Time-Series Data** (DOSM)

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/meridian-pahang-dashboard.git
   cd meridian-pahang-dashboard
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts ctivate
   ```

3. **Install dependencies:**
   Create a `requirements.txt` file (or use the one below) and run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure dataset alignment:**
   Create a folder named `datasets/` in the root directory and place your official DOSM Excel files inside:
   ```plaintext
   meridian/
   |
   |--- datasets/
   |    |--- LMR.xlsx
   |    |--- MultiDimensional_Poverty_Index.xlsx
   |    |--- Cost_of_Living_Indicators_2024.xlsx
   |    |--- SRU_rate_by_age.xlsx
   |
   |--- app.py
   |--- requirements.txt
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 📦 `requirements.txt` Reference

```plaintext
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.22.0
plotly>=5.10.0
openpyxl>=3.0.0
xgboost>=1.6.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
```

## 🛠️ Tech Stack

*   **Frontend/UI:** Streamlit
*   **Data Manipulation & Processing:** Pandas, NumPy
*   **Visualizations:** Plotly (Express & Graph Objects)
*   **Machine Learning (AI Pipeline):** Scikit-Learn (K-Means), XGBoost, Statsmodels

## 📄 License

This project is open-source and available under the MIT License.
