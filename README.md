CreditWise — Loan Approval Prediction Using Machine Learning
An exploratory, end-to-end binary classification project using applicant financial and personal information to study loan approval prediction. The work includes exploratory data analysis (EDA), missing-value treatment, categorical encoding, feature scaling, baseline model comparison, and additional feature-engineering experiments.
Project status: Experimental notebook with baseline and feature-engineered model results reported from screenshots of notebook outputs. The model comparisons below report both the initial baseline and the subsequent feature-engineering experiment on a single 80/20 train/test split. Preprocessing leakage and other limitations still need correction before claiming a robust benchmark. This is not a production loan-decision tool.

Business problem
In the project's fictional banking scenario, evaluating loan applications manually can be slow and inconsistent. The objective is to explore whether supervised machine learning can help classify applications as approved or not approved based on available applicant information.

Two errors matter differently: approving an application that would otherwise be rejected (a false approval) and rejecting an application that would otherwise be approved (a false rejection). Accuracy alone cannot capture the full trade-off.
Data
The notebook loads the dataset from data/loan_approval_data.csv and predicts Loan_Approved. The project data dictionary describes applicant income, co-applicant income, employment status, credit score, existing loans, debt-to-income (DTI) ratio, savings, collateral value, loan amount and term, loan purpose, property area, education, and other applicant characteristics.

Dataset size: The evaluation confusion matrices contain 200 test examples (139 labeled rejected, 61 labeled approved), corresponding to a 30.5% approval rate in this test split. The notebook specifies an 80/20 split, but total dataset size and missing-value counts have not been independently confirmed from the dataset. The code inspects them using .shape, .value_counts(), .info(), and .isnull().sum().
What I implemented
1. Data inspection and exploratory analysis
- Examined data types, missing values, unique-value counts, and descriptive statistics.
- Plotted the loan-approval class distribution.
- Visualized applicant and co-applicant income with histograms.
- Compared income, credit score, DTI ratio, and savings by approval class using boxplots.
- Compared credit-score and applicant-income distributions across approval outcomes.
- Computed correlations for numeric and encoded features and plotted a correlation heatmap.
2. Missing-value handling and encoding
- Used SimpleImputer(strategy="mean") for columns selected as float64.
- Used SimpleImputer(strategy="most_frequent") for object-type columns.
- Applied LabelEncoder to the target Loan_Approved and to Education_Level.
- Used OneHotEncoder(drop="first", handle_unknown="ignore") for Employment_Status, Marital_Status, Gender, Loan_Purpose, Property_Area, and Employer_Category.
- Combined encoded columns with the remaining features using pd.concat(..., axis=1).
Methodological note: The notebook currently fits imputation and encoding before splitting the data. For leakage-safe evaluation, these operations should be fitted on training data only using a pipeline or ColumnTransformer. The education encoding also needs an explicit, domain-justified order if treated as ordinal.
3. Baseline modeling
- Created an 80/20 train/test split using train_test_split(test_size=0.2, random_state=42).
- Fitted StandardScaler on the training features and transformed both training and test features.
- Trained and evaluated three supervised classification models:
  - Logistic Regression (max_iter=1000)
  - K-Nearest Neighbors (n_neighbors=7)
  - Gaussian Naive Bayes
- Used accuracy, precision, recall, F1-score, and confusion matrices to evaluate predictions.
4. Feature-engineering experiment
After the initial model comparison, I explored additional numerical feature representations:
- Applicant_Income_log = log1p(Applicant_Income)
- DTI_Ratio_sq = DTI_Ratio ** 2
- Credit_Score_sq = Credit_Score ** 2
I then constructed a revised feature matrix by dropping the original DTI_Ratio and Credit_Score columns, repeated the 80/20 split and scaling, and trained Logistic Regression, KNN, and Gaussian Naive Bayes again.

This is an experiment to compare feature representations; the shared notebook does not yet establish that these engineered features improved predictive performance.
Model evaluation: baseline vs. feature engineering
The notebook runs two experiments, each comparing Logistic Regression, K-Nearest Neighbors (K=7), and Gaussian Naive Bayes on an 80/20 split with random_state=42. The first uses the initial preprocessed features; the second uses additional engineered features. Metrics were copied from the displayed notebook evaluation outputs (200 test examples: 139 rejected, 61 approved), not independently rerun. Here, approval (Loan_Approved = 1) is the positive class.
Experiment 1 — Baseline models
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	88.00%	78.46%	83.61%	80.95%
KNN (K=7)	78.50%	70.45%	50.82%	59.05%
Gaussian Naive Bayes	85.50%	80.77%	68.85%	74.34%


Baseline confusion matrices (rows = actual, columns = predicted; class order 0 = rejected, 1 = approved):
Model	TN	FP	FN	TP
Logistic Regression	125	14	10	51
KNN (K=7)	126	13	30	31
Gaussian Naive Bayes	129	10	19	42


Experiment 2 — After feature engineering
The notebook creates Applicant_Income_log, DTI_Ratio_sq, and Credit_Score_sq, and removes original DTI_Ratio and Credit_Score from the updated feature matrix. It then retrains and evaluates all three algorithms.
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	87.00%	78.69%	78.69%	78.69%
KNN (K=7)	75.50%	64.29%	44.26%	52.43%
Gaussian Naive Bayes	86.50%	80.36%	73.77%	76.92%


Feature-engineered confusion matrices:
Model	TN	FP	FN	TP
Logistic Regression	126	13	13	48
KNN (K=7)	124	15	34	27
Gaussian Naive Bayes	128	11	16	45


What changed?
Model	Baseline accuracy	Engineered accuracy	Accuracy change	Baseline F1	Engineered F1
Logistic Regression	88.00%	87.00%	−1.0 pp	80.95%	78.69%
KNN (K=7)	78.50%	75.50%	−3.0 pp	59.05%	52.43%
Gaussian Naive Bayes	85.50%	86.50%	+1.0 pp	74.34%	76.92%


Key takeaways:
- The baseline Logistic Regression model is the best overall candidate on this test split, with 88.0% accuracy and 80.95% F1-score. It correctly identifies 51 of 61 actual approvals while making 14 false-approval predictions.
- Feature engineering was not uniformly beneficial. Logistic Regression and KNN lost accuracy and F1-score after the changes. Gaussian Naive Bayes improved from 85.5% to 86.5% accuracy and from 74.34% to 76.92% F1-score.
- Error types matter: Among engineered models, Naive Bayes had the fewest false approvals (11), while Logistic Regression identified more actual approvals (48 versus 45).
- The comparison evaluates several changes at once, so it does not isolate which individual feature caused each change. It also uses the same test split for repeated experimentation, which can influence subsequent model selection.
The above metrics are descriptive results from a single split and a pipeline that currently has preprocessing leakage. They should not be interpreted as deployment-ready or as proof that feature engineering improves generalization.
Technologies
- Language: Python
- Data wrangling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine learning: Scikit-learn
- Environment: Jupyter Notebook / VS Code
Repository structure
credit_wise/
├── data/
│   └── loan_approval_data.csv   # Only if redistribution is permitted
├── main.ipynb                   # EDA, preprocessing, experiments, evaluation
├── README.md                    # Project documentation
├── requirements.txt             # Project dependencies
└── .gitignore                   # Local/private file exclusions
How to run
git clone https://github.com/pk-patel-13/creditwise-ML.git
cd creditwise-ML
python -m venv .venv
Activate the environment (Windows PowerShell):
.\.venv\Scripts\Activate.ps1
Install dependencies and open main.ipynb in Jupyter or VS Code:
pip install -r requirements.txt
Place an authorized copy of the dataset at data/loan_approval_data.csv. Run the notebook cells in order. Note: Before treating the results as reproducible final benchmarks, address the preprocessing and identifier issues listed below.
Limitations and next steps
1. Prevent preprocessing leakage: Fit imputers and encoders on training data rather than the complete dataset, ideally with Pipeline and ColumnTransformer.
2. Remove identifiers: The notebook calls df_raw.drop(columns="Applicant_ID") without assigning the result or using inplace=True; confirm the identifier is actually excluded from the feature matrix.
3. Check categorical/ordinal handling: Confirm target-label mapping, specify education ordering deliberately, and inspect which numeric columns are included in imputation.
4. Stratify the split: Consider stratify=y so the class proportions are preserved in training and testing.
5. Save reproducible outputs: Save both baseline and engineered evaluation outputs in the notebook, preserve dataset provenance, and rerun with leakage-safe validation before making strong generalization claims.
6. Strengthen validation: Compare cross-validation results, inspect class balance, and evaluate fairness and error costs before considering deployment.
This is an educational classification experiment based on a fictional lending scenario. Approval labels can encode historical decision patterns and should not be treated as inherently fair or correct. The models are not suitable for real credit decisions without additional validation, governance, and compliance review.
Author
Param Patel
B.S. Data Science and Artificial Intelligence — Florida International University
GitHub repository