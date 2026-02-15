 **Bank Marketing Subscription Prediction – ML & Streamlit Deployment**

 
 
 **a. Problem Statement**

The objective of this project is to build multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on data from direct marketing campaigns of a Portuguese banking institution.

The project includes:

- Implementation of multiple classification algorithms

- Model evaluation using comprehensive performance metrics

- Development of an interactive Streamlit web application

- Deployment-ready ML workflow

  

**b. Dataset Description**

- **Dataset Name:** Bank Marketing Dataset

- **Source:** UCI Machine Learning Repository

- **Direct Link:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

- **Instances:** 45,211

- **Features:** 16 input features + 1 target variable

- **Feature Types:**

- **Categorical Features:** job, marital, education, default, housing, loan, contact, month, poutcome

- **Numerical Features:** age, balance, day, duration, campaign, pdays, previous

- **Target Variable:** y (Yes/No – whether client subscribed to term deposit)

The dataset contains no missing values. 

**The full dataset (bank-full.csv) and the test dataset (test_data.csv) both are uploaded in the repository.**

This dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is to predict whether a client will subscribe to a term deposit or not.

- **Citation:**

Moro, S., Cortez, P., & Rita, P. (2014).
A data-driven approach to predict the success of bank telemarketing.
Decision Support Systems.

**Data Preprocessing Steps:**

1.Converted target variable (yes → 1, no → 0).

2.Performed One-Hot Encoding on categorical variables.

3.Ensured column alignment between training and test data.

4.Applied StandardScaler for feature normalization.

5.Split dataset into training (80%) and testing (20%) sets.

6.Saved feature column metadata for deployment consistency.



**c. Models Implemented**

**The following classification models were implemented:**

1.Logistic Regression

2.Decision Tree

3.K-Nearest Neighbors (KNN)

4.Gaussian Naive Bayes

5.Random Forest

6.XGBoost

**Evaluation Metrics Used**

Each model was evaluated using:

1.Accuracy

2.AUC Score

3.Precision

4.Recall

5.F1 Score

6.Matthews Correlation Coefficient (MCC)

*These metrics provide a comprehensive understanding of performance, especially considering class imbalance.


**Model Performance Comparison**
| Model               | Accuracy   | AUC        | Precision  | Recall     | F1 Score   | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9012     | 0.9054     | 0.6440     | 0.3488     | 0.4525     | 0.4264     |
| Decision Tree       | 0.8777     | 0.7135     | 0.4783     | **0.4991** | **0.4884** | 0.4191     |
| KNN                 | 0.8936     | 0.8084     | 0.5860     | 0.3091     | 0.4047     | 0.3742     |
| Naive Bayes         | 0.8639     | 0.8088     | 0.4282     | 0.4877     | 0.4560     | 0.3797     |
| Random Forest       | 0.9045     | **0.9272** | 0.6554     | 0.3866     | 0.4863     | **0.4561** |
| XGBoost             | **0.9047** | 0.9240     | **0.6667** | 0.3705     | 0.4763     | 0.4510     |




| Category                 | Best Model              | Metric Value | Interpretation                                  |
| ------------------------ | ----------------------- | ------------ | ----------------------------------------------- |
| Highest Accuracy         | XGBoost                 | 0.9047       | Best overall prediction performance             |
| Highest Precision        | XGBoost                 | 0.6667       | Most reliable positive predictions              |
| Highest AUC              | Random Forest           | 0.9272       | Best class discrimination ability               |
| Highest MCC              | Random Forest           | 0.4561       | Best balanced performance on imbalanced dataset |
| Highest Recall           | Decision Tree           | 0.4991       | Identifies maximum actual subscribers           |
| Overall Strongest Models | Random Forest & XGBoost | —            | Ensemble methods outperform individual models   |


| ML Model Name            | Observation about Model Performance                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved strong overall performance with high AUC and good precision, but exhibited relatively lower recall, indicating difficulty in identifying all positive cases. |
| Decision Tree            | Achieved the highest recall among all models, effectively identifying more positive instances, but showed lower AUC due to potential overfitting.                     |
| KNN                      | Delivered moderate performance but struggled in high-dimensional feature space after one-hot encoding, resulting in comparatively lower recall and MCC.               |
| Naive Bayes              | Provided reasonable recall but lower precision due to strong independence assumptions among features, limiting its ability to capture complex relationships.          |
| Random Forest (Ensemble) | Achieved the highest AUC and MCC, indicating superior balanced performance and strong class discrimination ability.                                                   |
| XGBoost (Ensemble)       | Achieved the highest accuracy and precision, demonstrating strong predictive capability and robustness.                                                               |
| Overall Conclusion       | Ensemble models (Random Forest and XGBoost) outperformed individual models, providing the most reliable and balanced predictions for this dataset.                    |




**d.Streamlit Application Features**

The deployed Streamlit app includes:

- CSV dataset upload option (a dataset download option was provided, from which the test_data.csv can be downloaded to the computer.It can later be uploaded through 'Browse Files' option to see the results)

- Built-in test dataset option (second option without downloading or uploading of test dataset)

- Model selection dropdown (default: None)

- Data source selection (default: Upload CSV)

- Display of all 6 evaluation metrics

- Compact confusion matrix visualization

- Classification report visualization

- Sample test dataset download button




**e.How to Run Locally**

1.Clone the repository

2.Install dependencies:

pip install -r requirements.txt

3.Run Streamlit:

streamlit run app.py




**f.Repository Contents**

- 2025aa05860_ml_assignment2.ipynb – Model implementation & evaluation

- app.py – Streamlit application

- requirements.txt – Deployment dependencies

- test_data.csv – Sample test dataset

- model_files/ – Saved trained models and scaler




**g.Author**
Viyyuri P Praveen Kumar,
2025AA05860,
Machine Learning Assignment 2,
WILP, BITS Pilani.


