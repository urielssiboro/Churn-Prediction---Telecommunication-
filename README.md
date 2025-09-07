# Customer Churn Prediction

This project builds and evaluates machine learning models to predict customer churn for a telecom company.  
Churn = whether a customer leaves the service provider.  
The main business goal is to minimize False Negatives (FN) — customers who churn but are predicted to stay — since losing customers is far more costly than offering retention benefits unnecessarily.

---

## Project Structure
- `data/` → raw and processed datasets  
- `notebooks/` → Jupyter notebooks for EDA, preprocessing, and modeling  
- `models/` → saved models and pipelines (`joblib`)  
- `README.md` → project overview  

---

## Dataset
The dataset contains ~4,800 customer records with the following features:

- **Dependents**: Whether the customer has dependents  
- **Tenure**: Number of months the customer has stayed  
- **OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport**: Subscription to additional services  
- **InternetService**: Type of internet service (No, DSL, Fiber optic)  
- **Contract**: Contract type (Month-to-month, One year, Two year)  
- **PaperlessBilling**: Whether the customer uses paperless billing  
- **MonthlyCharges**: Monthly service charges  
- **Churn**: Target variable (Yes = churn, No = stay)  

---

## Preprocessing
- Ordinal encoding applied to service/contract features with a natural order (e.g., `No < DSL < Fiber`).  
- One-hot encoding used for nominal variables (e.g., `PaperlessBilling`).  
- RobustScaler used for numeric variables (`tenure`, `MonthlyCharges`).  
- Resampling (ROS, RUS, NearMiss, SMOTE) tested to address class imbalance.  
- Pipeline + ColumnTransformer ensure preprocessing and modeling are reproducible and consistent.

---

## Models Tested
- Logistic Regression (Ridge, Lasso, Elastic Net)  
- Tree-based models: Decision Tree, Random Forest, Extra Trees  
- Boosting models: XGBoost, LightGBM  

Each model was evaluated with and without resampling techniques.  

---

## Evaluation Metrics
To align with the business problem (catch churners and minimize FN), we used:

- **Recall (PRIORITY)**: Most important metric, measures how many churners are caught.  
- **Precision**: Controls how many false alarms (unnecessary retention offers) are made.  
- **F1-score**: Balances precision and recall.  
- **ROC-AUC**: Overall ability to separate churn vs no-churn.  
- **PR-AUC**: More relevant for imbalanced data, measures trade-off between recall and precision.  

---

## Key Findings
- Best model: Logistic Regression (Lasso ) with recall ~0.93 and PR-AUC ~0.63.  
- Lasso chosen for automatic feature selection and interpretability.  
- Threshold tuning improves recall/precision trade-off.  
- Tree/boosting models were competitive but more complex and prone to overfitting on the small dataset.  

---

## Limitations
- Small dataset (~4.8k rows) → complex models risk overfitting.  
- Target imbalance (~27% churn) → requires weighting or resampling.  
- Ordinal encoding assumes order that may not always reflect true business impact.  
- New/unseen categories in test data are ignored by OneHotEncoder.  
- Trade-off: high recall comes at the expense of precision.  

---

## Deployment
- Model pipeline (preprocessing + model) saved with `joblib`.  
- For deployment:
  - Input data must match training schema (columns, order, data types).  
  - Target mapping (`Yes/No → 1/0`) must remain consistent.  
- Logistic Regression is lightweight and deployment-friendly.  
- Threshold should be tuned based on desired recall/precision trade-off.  

---

## Next Steps
- Add more customer behavior features (usage, payments, complaints).  
- Perform advanced hyperparameter tuning for boosting models.  
- Explore cost-sensitive metrics (e.g., assigning business cost to FN vs FP).  
- Test model robustness on larger and more diverse datasets.  
