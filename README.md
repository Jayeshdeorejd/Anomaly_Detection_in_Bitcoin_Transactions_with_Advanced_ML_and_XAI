```markdown
# Anomaly Detection in Bitcoin Blockchain Transactions with Advanced ML and XAI
---

## Project Overview

Blockchain transactions, especially Bitcoin, are highly imbalanced: of over 30 million transactions, only 108 are labelled fraudulent. This project implements and evaluates multiple machine‑learning approaches—single and ensemble tree‑based models—combined with novel sampling techniques to detect anomalous Bitcoin transactions accurately while retaining interpretability through XAI (SHAP) and decision‑rule extraction.

---

## Key Features

- **Data Balancing**  
  - Traditional under‑sampling: Random Under‑Sampling (RUS), NearMiss  
  - Novel XGBCLUS under‑sampling algorithm  
  - Over‑sampling: SMOTE, ADASYN  
  - Hybrid: SMOTEENN, SMOTETomek

- **Classification Models**  
  - Single tree‑based: Decision Tree, Random Forest, Gradient Boosting, AdaBoost  
  - Ensemble: Hard & Soft Voting, Stacked Ensemble with Logistic Regression meta‑classifier  

- **Explainability & Interpretability**  
  - Global & local SHAP analysis to rank and visualize feature impact (e.g., `total_btc` is most influential) :contentReference[oaicite:2]{index=2}  
  - Extraction of human‑readable “anomaly rules” from decision trees (e.g., `if total_btc > 96.6 & in_btc > 236.2 then Anomalous (98–100% confidence)`) 

- **Deployment**  
  - Final stacked ensemble model hosted on Google Colab for real‑time/batch inference.

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

##  Installation

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

##  Usage

## ▶️ Usage

1. **Access the data**  
   The raw dataset is stored on Google Drive. You can download it here: **[Drive Link]([https://drive.google.com/your-data-link](https://drive.google.com/file/d/1kwhkOTtzikQFhuRBF_RwgCAKdOa0AeTV/view?usp=drive_link))**

2. **Access the report**  
   The detailed project report is available via a secured Google Drive link. Please request access here: **[Request Report Access]([https://drive.google.com/your-report-link](https://docs.google.com/document/d/1-LblGY6NcUy_iQvslUN9NwxlhLBZh8on/edit?usp=sharing&ouid=113317600828046355730&rtpof=true&sd=true))**

3. **Prepare the data**  
   - After downloading, place the file in `data/bitcoin_transactions.csv`.

4. **Open & run the notebook**  
   ```bash
   jupyter notebook notebooks/Final_codes.ipynb


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
