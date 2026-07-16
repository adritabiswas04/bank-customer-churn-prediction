# Predictive Modeling and Risk Scoring for Bank Customer Churn

This submission implements the project exactly around the required objectives: churn prediction, probability-based risk scoring, churn-driver analysis, explainability, and a Streamlit dashboard for what-if analysis.

## Dataset

- Source file: `data/European_Bank.csv`
- Rows: 10,000
- Columns: 14
- Target variable: `Exited`
- Churn rate: 20.37%
- Non-informative fields dropped for modeling: `CustomerId`, `Surname`, `Year`

## Project structure

```text
bank_churn_submission/
├── data/
│   └── European_Bank.csv
├── figures/
│   ├── EDA charts
│   ├── ROC curves
│   ├── confusion matrix
│   └── feature importance
├── models/
│   ├── bank_churn_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── Research_Paper_Bank_Churn.docx
│   ├── Research_Paper_Bank_Churn.pdf
│   ├── Executive_Summary_Government_Stakeholders.docx
│   └── Executive_Summary_Government_Stakeholders.pdf
├── Bank_Churn_Project_Notebook.ipynb
├── streamlit_app.py
├── model_metrics.csv
├── feature_importance.csv
├── risk_scored_customers.csv
├── requirements.txt
└── README.md
```

## Modeling approach

Required methodology implemented:

1. Data preprocessing
   - Missing value check
   - Removal of `CustomerId`, `Surname`, and constant `Year`
   - One-hot encoding for `Geography` and `Gender`
   - Numerical scaling
2. Feature engineering
   - `BalanceSalaryRatio`
   - `ProductDensity`
   - `EngagementProductInteraction`
   - `AgeTenureInteraction`
3. Train-test strategy
   - Stratified 80/20 train-test split
   - 5-fold cross-validation for best model ROC-AUC
4. Model development
   - Logistic Regression baseline
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost, if installed
5. Evaluation
   - Accuracy, precision, recall, F1-score, ROC-AUC
   - Confusion matrix
   - ROC curve comparison
6. Explainability
   - Feature importance ranking
   - Partial dependence plots
   - SHAP summary bar chart
7. Streamlit dashboard
   - Customer churn risk calculator
   - Probability distribution visualization
   - Feature importance dashboard
   - What-if scenario simulator
   - Portfolio risk table

## Best model result

Best model selected by ROC-AUC and F1-score: **XGBoost**

| Metric | Value |
|---|---:|
| Accuracy | 0.8695 |
| Precision | 0.7897 |
| Recall | 0.4889 |
| F1-Score | 0.6039 |
| ROC-AUC | 0.8691 |

Chosen risk-flag threshold: **0.2523**

## Run the Streamlit dashboard

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Main output files

- `risk_scored_customers.csv`: customer-level churn probability, risk score, risk band, and churn flag.
- `feature_importance.csv`: ranked churn-driver importance.
- `model_metrics.csv`: model comparison table.
- `reports/Research_Paper_Bank_Churn.docx`: full research paper.
- `reports/Executive_Summary_Government_Stakeholders.docx`: policy-friendly executive summary.
