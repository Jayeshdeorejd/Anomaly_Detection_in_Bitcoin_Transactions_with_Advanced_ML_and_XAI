```markdown
# Anomaly Detection in Bitcoin Blockchain Transactions with Advanced ML and XAI
---

## Project Overview

Blockchain transactions, especially Bitcoin, are highly imbalanced: of over 30 million transactions, only 108 are labelled fraudulent :contentReference[oaicite:0]{index=0}. This project implements and evaluates multiple machine‑learning approaches—single and ensemble tree‑based models—combined with novel sampling techniques to detect anomalous Bitcoin transactions accurately while retaining interpretability through XAI (SHAP) and decision‑rule extraction.

---

## Key Features

- **Data Balancing**  
  - Traditional under‑sampling: Random Under‑Sampling (RUS), NearMiss  
  - Novel XGBCLUS under‑sampling algorithm :contentReference[oaicite:1]{index=1}  
  - Over‑sampling: SMOTE, ADASYN  
  - Hybrid: SMOTEENN, SMOTETomek

- **Classification Models**  
  - Single tree‑based: Decision Tree, Random Forest, Gradient Boosting, AdaBoost  
  - Ensemble: Hard & Soft Voting, Stacked Ensemble with Logistic Regression meta‑classifier  

- **Explainability & Interpretability**  
  - Global & local SHAP analysis to rank and visualize feature impact (e.g., `total_btc` is most influential) :contentReference[oaicite:2]{index=2}  
  - Extraction of human‑readable “anomaly rules” from decision trees (e.g., `if total_btc > 96.6 & in_btc > 236.2 then Anomalous (98–100% confidence)`) :contentReference[oaicite:3]{index=3}  

- **Deployment**  
  - Final stacked ensemble model hosted on Google Colab for real‑time/batch inference :contentReference[oaicite:4]{index=4}  

---

##  Repository Structure

```

.
├── README.md
├── data/
│   └── bitcoin\_transactions.csv         # raw/processed dataset (if included)
├── notebooks/
│   └── Final\_codes.ipynb                # Jupyter notebook with all code
├── report/
│   └── Group 3 Report.pdf               # full project write‑up
├── requirements.txt                     # Python dependencies
└── LICENSE

````

---

## ⚙ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/bitcoin-anomaly-detection.git
   cd bitcoin-anomaly-detection
````

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **`requirements.txt`** should include (at minimum):

   ```
   pandas
   numpy
   scikit-learn
   imbalanced-learn
   xgboost
   shap
   matplotlib
   seaborn
   ```

---

## ▶ Usage

1. **Prepare the data**

   * Place the cleaned dataset in `data/bitcoin_transactions.csv`.

2. **Open & run the notebook**

   ```bash
   jupyter notebook notebooks/Final_codes.ipynb
   ```

   * Follow each cell to load data, apply sampling techniques, train models, run SHAP analyses, and extract rules.

3. **Interactive deployment**

   * The final stacked ensemble is demonstrated in Colab; simply open \[Colab link] and run.

---

##  Results & Findings

* **Under‑sampling (XGBCLUS)** achieved the highest **TPR** (up to 0.83) with relatively low **FPR** (\~0.27), outperforming RUS and NearMiss in ROC‑AUC scores (≈0.83) .
* **Over‑sampling** (SMOTEENN) yielded the lowest FPR (\~0.04) but at the cost of lower TPR (\~0.44) .
* **Ensemble Stacking** with XGBCLUS sampling delivered the best overall balance:

  * **Accuracy**: \~0.82
  * **TPR**: \~0.83
  * **FPR**: \~0.18
  * **ROC‑AUC**: \~0.80 .
* **SHAP** identified `total_btc`, `mean_in_btc`, and `in_btc` as the top predictors; `out_btc` and `mean_out_btc` were least influential .
* **Anomaly rules** (e.g., thresholds around 25 BTC to flag anomalies) provide human‑readable decision criteria .

---
