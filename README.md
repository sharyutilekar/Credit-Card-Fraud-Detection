# 💳 Credit Card Fraud Detection

Machine learning project to detect fraudulent credit card transactions on a highly imbalanced dataset (only **0.17%** fraud), comparing Logistic Regression and Random Forest and tuning the decision threshold for the best precision/recall trade-off.

**Final model:** Random Forest + tuned threshold → **87.8% precision, 75.8% recall, 0.81 F1** on an unseen test set (72 of 95 frauds caught with only 10 false alarms out of 56,651 legitimate transactions).

---

## 📌 Problem Statement

Banks lose billions to card fraud every year. A good fraud model has to catch as much fraud as possible **without** blocking large numbers of genuine customers. Because fraud is extremely rare, plain accuracy is misleading (a model that always predicts "not fraud" gets 99.8% accuracy), so this project is evaluated with **PR-AUC, precision, recall and F1**.

## 📂 Dataset

- Source: [Kaggle – Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions made by European cardholders over two days
- 30 features: `Time`, `Amount`, and `V1–V28` (PCA-anonymised for confidentiality)
- Target: `Class` (1 = fraud, 0 = legitimate)
- After removing 1,081 duplicate rows: **283,726 transactions, 473 frauds (0.17%)**

> The dataset is not included in this repo because of its size. Download `creditcard.csv` from Kaggle and place it in the project root.

## 🔧 Approach

1. **Data cleaning** – checked missing/infinite values, removed duplicates.
2. **Train/test split** – 80/20, stratified so both sets keep the same fraud rate. The test set is never used for tuning.
3. **Preprocessing (inside a scikit-learn `Pipeline`, so no data leakage)**
   - `Time` and `Amount` (not PCA-scaled): median imputation → custom IQR outlier clipper → Yeo-Johnson power transform (`Amount` has skewness of ~14.4)
   - `V1–V28`: passed through unchanged (already PCA outputs)
4. **Handling imbalance** – `class_weight="balanced"` (Logistic Regression) and `"balanced_subsample"` (Random Forest) instead of resampling.
5. **Modelling** – GridSearchCV with 5-fold StratifiedKFold, optimised for **PR-AUC (average precision)**.
6. **Threshold tuning** – the decision threshold was chosen on **cross-validated predictions of the training set** (maximising F1), then applied once to the test set.

## 📊 Results (held-out test set: 56,746 transactions, 95 frauds)

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.7557 | 0.6705 | 0.9620 | 0.0551 | 0.8737 |
| **Random Forest** | **0.8169** | **0.7977** | 0.9579 | **0.8690** | 0.7684 |

**Random Forest after threshold tuning (0.535):**

| Metric | Default (0.5) | Tuned (0.535) |
|---|---|---|
| Precision | 0.8690 | 0.8780 |
| Recall | 0.7684 | 0.7579 |
| F1 | 0.8156 | 0.8136 |

Confusion matrix (tuned): 56,641 TN · 10 FP · 23 FN · 72 TP

### Charts

![Class imbalance](images/class_imbalance.png)
![Confusion matrix](images/confusion_matrix.png)
![Model comparison](images/model_comparison.png)

## 💡 Key Insights

- **Accuracy is misleading here.** Logistic Regression scored 97.5% accuracy but only 5.5% precision: it flagged 1,422 genuine transactions as fraud to catch 83 real ones.
- **Random Forest is far more practical.** It has much higher precision (87% vs 5.5%) and a better PR-AUC (0.80 vs 0.67) with a modest drop in recall, meaning far fewer false alarms for customers and fraud analysts.
- **Business trade-off:** Logistic Regression suits a "catch everything" strategy (recall 87%) where manual review is cheap. Random Forest suits a strategy where customer friction is costly.
- ROC-AUC looked excellent (~0.96) for both models even though their real-world usefulness differed sharply, which is why PR-AUC was used for model selection.

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── credit_card_fraud_detection.ipynb   # full analysis & modelling
├── images/                             # charts used in this README
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
# download creditcard.csv from Kaggle into this folder
jupyter notebook credit_card_fraud_detection.ipynb
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## ⚠️ Limitations & Future Work

- Random (stratified) split was used; a **time-based split** would better simulate real deployment.
- Only 95 fraud cases in the test set, so metrics have noticeable variance.
- Try gradient boosting (XGBoost / LightGBM), SMOTE or cost-sensitive learning.
- Add model explainability (SHAP) and a cost-based threshold using real fraud loss vs. review cost.
- Wrap the model in a simple API/Streamlit app for demo purposes.

## 👤 Author

**Your Name** · [LinkedIn](https://www.linkedin.com/in/your-profile) · [Email](mailto:your@email.com)
