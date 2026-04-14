# Customer Conversion Prediction

A machine learning system that predicts the likelihood of a bank customer subscribing to a term deposit - helping marketing teams prioritize outreach, reduce campaign costs, and improve conversion rates.

**[Live Demo ->](https://customer-conversion-prediction-eyqqkcet3eqycapj5qvswn.streamlit.app/)**

## Business Problem

Marketing campaigns are expensive. Contacting every customer indiscriminately wastes resources and risks alienating those unlikely to convert. This project builds a predictive model that scores each customer's conversion likelihood before outreach, enabling teams to focus effort where it matters most.

## Dataset

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Size:** 41,188 records, 20 features
- **Target:** Whether a customer subscribed to a term deposit (yes/no)
- **Class imbalance:** ~11% positive class - handled via `scale_pos_weight` and class weights

## Approach

### 1. Exploratory Data Analysis
- Class distribution, age distribution by outcome, subscription rate by job type and month
- Correlation heatmap to identify multicollinearity
- Call duration vs subscription outcome analysis

### 2. Data Cleaning & Preprocessing
- Removed zero-duration calls (no meaningful interaction occurred)
- Replaced `pdays=999` (never contacted) with 0 for interpretability
- Dropped highly correlated macro features (`emp.var.rate`, `nr.employed`) - retained `euribor3m`
- Label encoding for binary features, one-hot encoding for categorical features

### 3. Model Development
Four models trained and compared, progressing from baseline to best:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.861 | 0.447 | 0.885 | 0.594 | 0.936 |
| Random Forest | 0.913 | 0.701 | 0.415 | 0.522 | 0.943 |
| XGBoost | 0.847 | 0.425 | 0.941 | 0.585 | 0.947 |
| **LightGBM** | **0.857** | **0.442** | **0.939** | **0.601** | **0.951** |

XGBoost and LightGBM tuned with `RandomizedSearchCV` optimizing for **recall** - missing a likely subscriber (false negative) has higher business cost than a false positive.

### 4. Threshold Tuning
Default classification threshold of 0.5 is suboptimal for imbalanced data. Optimal threshold found at **0.75** by maximizing F1 score across the precision-recall tradeoff curve.

### 5. Explainability (SHAP)
SHAP values used to explain both global feature importance and individual predictions. Top drivers:
- **Call duration** - strongest predictor; longer calls signal genuine engagement
- **Euribor 3-month rate** - customers subscribe more in lower interest rate environments
- **Contact month** - March, September, October, December show higher conversion rates
- **Consumer confidence & price indices** - macro conditions influence customer receptiveness

## Key Findings

- **LightGBM** achieves the best AUC-ROC (0.951) with ~94% recall on the positive class
- **Duration leakage caveat:** Call duration is only known after the call ends, making it unavailable for pre-call scoring. In a production pre-call scoring system, this feature should be excluded and the model retrained. It is retained here as the dataset is intended for post-call analysis
- **Economic indicators** matter significantly - campaigns perform better during periods of low Euribor rates and higher consumer confidence

## Streamlit App

The live app allows users to input customer features and receive:
- Conversion probability score
- Likely/Unlikely verdict at the optimal 0.75 threshold
- Contextual business insight cards
- SHAP waterfall plot explaining exactly why the model made that prediction

**[Try it live ->](https://customer-conversion-prediction-eyqqkcet3eqycapj5qvswn.streamlit.app/)**

## Project Structure

```
customer-conversion-prediction/
├── customer_conversion_prediction.ipynb  # Full analysis and modelling
├── app.py                                # Streamlit deployment app
├── requirements.txt                      # Python dependencies
├── best_model.pkl                        # Trained LightGBM model
├── feature_columns.pkl                   # Feature column names for alignment
├── best_threshold.pkl                    # Optimal classification threshold
├── label_encoders.pkl                    # Fitted label encoders for preprocessing
└── bank-additional-full.csv             # Source dataset
```

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Frederama/customer-conversion-prediction.git
cd customer-conversion-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## Tech Stack

- **Modelling:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **App:** Streamlit
- **Data:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
