# Introduction
**GradScope** is a predictive system designed to estimate a student’s chances of being admitted to graduate school based on standardized scores, academic records, and research experience.

**Key Objectives:**
- Build an interpretable ML model for admission prediction
- Identify the most influential features in the decision-making process
- Provide transparency and insights to applicants and institutions

# Problem Statement
Graduate admissions can often feel like a black box to applicants. Decisions are influenced by multiple academic and personal factors, yet the criteria for acceptance can seem unclear. With thousands of students applying each year, a data-driven approach to predicting admissions outcomes can help both applicants and institutions better understand the selection process. 
This project aims to develop machine learning models to predict whether a student will be admitted into a graduate program based on their academic profile, including standardized test scores, university rating, CGPA and research background.

# Methodology
- Data Loading
- Data exploration and cleaning
- Feature engineering
- Model selection and Training (XGBoost)
- Evaluation using F1 Score and cross validation
- Feature importance
- App deployment on Streamlit

![image](https://github.com/user-attachments/assets/52cab667-f690-456c-af7b-0785aa6c7c31)

# ETL Process (Extract → Explore → Clean&Transform)
The GradScope project uses a streamlined ETL (or rather ELT) process due to the static, structured nature of the dataset.

1.  #### Load Data into DataFrame
The dataset is stored locally in CSV format. It is read using pandas into a DataFrame.
Data is gotten from Zindi via the competition page: [Data](https://zindi.africa/competitions/from-scores-to-seats-the-grad-school-ml-challenge/data)
The dataset includes features such as GRE, TOEFL, CGPA, SOP (Statement of Purpose), LOR (Letter of Recommendation), University Rating, Age, Research, and Location.

2.  #### Exploratory Data Analysis (EDA)
The EDA phase was essential in understanding feature distributions and identifying patterns within the dataset. It laid the foundation for data cleaning and modelling
decisions.

3. #### Data Cleaning
Key cleaning actions were performed based on the observation from EDA to prepare data for modeling:
- Replace CGPA values>10 with 10,
- TOEFL Score>120 with 120, 
- SOP>5 with 5
- LOR>5 with 5, 
- GRE Score>340 with 340
- Replace negative and zero values in SOP and LOR with 1
- Fill missing values

4. #### Feature Engineering
In addition to cleaning, GradScope applies domain-informed feature engineering to enhance model performance. Below are the new features created and their rationale:
- academic_strength: A weighted average of GRE and TOEFL scores. GRE was found to have higher importance than TOEFL in influencing admission. This feature gives the overall academic preparedness with an emphasis on GRE.
- Gre_university: Interaction between GRE score and university rating. High GRE scores may matter more when applying to highly ranked universities. This feature captures the synergy between a candidate’s performance and the selectivity of the institution.
- Research_strength: A composite measure that combines Research Experience with CGPA, SOP, LOR, and GRE.

# Model Development
GradScope leverages XGBoost, A gradient-boosted decision tree algorithm known for handling tabular data well and offering superior performance in classification tasks.
Why XGBoost:
- The relationships between features and admission status are non-linear, justifying the choice of XGBoost over simpler linear models.
- Class imbalance exists (66.5% admitted vs. 33.5% not admitted), requiring attention during training. Regularization helps prevent overfitting


**Class Imbalance Handling**
The dataset had 66.5% admitted and 33.5% not admitted. To mitigate these bias weights were computed and passed into the model to ensure fairness in prediction across both classes.

![image](https://github.com/user-attachments/assets/2ecc77cc-c021-49d4-a397-84ef487c4478)

```python
def calculate_class_weights(y):

    #Function to calculate class weights based on class distribution in the target variable.
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight

    return class_weights
```
**Data Split**
- 90% for training
- 10% held-out validation
- stratify=target maintains label balance

**Stratified K-Fold Cross-Validation**
Cross-validation was used to assess robustness and generalizability:
- 5 folds
- Mean F1-score used as the evaluation metric across splits
```
Fold 1/5
F1 Score: 0.9046
Fold 2/5
F1 Score: 0.9085
Fold 3/5
F1 Score: 0.9068
Fold 4/5
F1 Score: 0.8902
Fold 5/5
F1 Score: 0.8974
Mean F1 Score: 0.9015
Mean F1-Score across folds (XgBoost): 0.9015
```
**Feature Importance**

![image](https://github.com/user-attachments/assets/20aba178-35c7-4b98-b88e-786469799561)

Top Features Identified:
- Gre_university
- Research
- Research_strength
- GRE Score
- academic_strength.
These engineered features played a crucial role in improving model performance by capturing composite effects and interactions.

**Model Evaluation**
The model was evaluated on the 10% split and the results obtained is shown below:
```
F1 Score: 0.8940
AUC Score: 0.8396
Accuracy Score: 0.8583
```
These scores show excellent precision-recall trade-off, good overall correctness, and strong discriminatory power across threshold levels.
**Confusion Matrix**
A confusion matrix is a powerful evaluation tool that helps you visualize how well your classification model is performing by comparing actual vs. predicted labels. In GradScope, False Negatives could hurt applicants who were actually strong candidates, while False Positives could waste admission officers’ time. The confusion matrix lets you quantify both.

![image](https://github.com/user-attachments/assets/4ccc8f9e-877d-417a-89a9-7aaa3527d4c0)

- The model correctly predicted  309 out of 360 cases.
- Only 59 cases were misclassified: 26 students wrongly predicted as admitted and 25 missed chances.
The Confusion Matrix showed good separation between admitted and not admitted classes.

**ROC Curve**
The AUC (Area Under Curve) summarizes performance — a perfect classifier scores 1.0, a random one 0.5.

![image](https://github.com/user-attachments/assets/fa0cceba-7d18-49fd-a5a1-e3815d4b273f)

- The model has an 83.9% probability of correctly ranking a admitted applicant higher than a non-admitted one.
- confirms good (though not perfect) ability to distinguish between admitted and rejected applicants

The model achieved strong performance with an 89.4% F1-score, demonstrating excellent balance between precision and recall in classifying admission status fpr applicants. An 85.83% accuracy indicates robust overall prediction correctness, while the 83.96% AUC-ROC score confirms good (though not perfect) ability to distinguish between admitted and rejected applicants. Together, these metrics suggest the model is highly reliable for identifying successful candidates while maintaining reasonable discrimination across probability thresholds.

This solution placed GradSchoolers 5th on the leaderboard
![image](https://github.com/user-attachments/assets/ea6f50ca-5041-4d73-b919-a3b64becc126)
For more information about the solution, reach out: [AI_Maven](https://zindi.africa/users/AI_Maven), [flibbert_debola](https://zindi.africa/users/flibbert_debola)

# GradScope Development and Usage
#### Overview
GradScope was built using streamlit, a python library for creating interactive web applications. This section presents a video that describes how to utilize the app to make predictions.

Some Images of GradScope

<img width="960" alt="Gradscope overview" src="https://github.com/user-attachments/assets/5bda5842-3b6a-45db-b783-1a8138c100fa" />

<img width="960" alt="GradScope showcase 3 - f" src="https://github.com/user-attachments/assets/9824752a-7279-4966-b487-1292aa23e50f" />

<img width="959" alt="GradScope showcase 4 -f" src="https://github.com/user-attachments/assets/2158c894-4d16-44f5-bee6-5fbc018e83e9" />

<img width="960" alt="GradScope showcase 5-f" src="https://github.com/user-attachments/assets/1698b5a0-9119-49e6-a13e-614b88e63ff9" />

<img width="775" alt="GradScope showcase" src="https://github.com/user-attachments/assets/38f8ef15-5c6b-4eec-a933-81e6ae96817f" />

<img width="662" alt="Gradscope showcase2" src="https://github.com/user-attachments/assets/3707baa2-db39-470c-92bf-c4c9952337cf" />

Below is a video that shows how to utilize GradScope

[GradScope Video](https://github.com/obinna-Muonanu/GradScope/blob/main/streamlit-GradScope-2025-04-08-11-04-34.webm)
