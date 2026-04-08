# Employee Attrition Prediction

## Overview
This project analyzes employee attrition using the IBM HR Analytics Employee Attrition dataset and compares multiple machine learning models built in R. The objective is to identify the main factors associated with employee turnover and support retention-focused business decisions through predictive analytics.

## Business Problem
Employee attrition creates hiring costs, training costs, productivity loss and operational disruption. The goal of this project is to understand which employee attributes are most associated with attrition and compare predictive models that can help HR teams identify higher-risk cases earlier.

## Dataset
- Source: IBM HR Analytics Employee Attrition dataset
- Records: 1470 employees
- Features: 35 columns in the original dataset
- Data quality checks performed: dataset dimensions, missing values, duplicates, descriptive statistics and summary review

## Tools and Libraries
R, readr, dplyr, psych, ggplot2, caret, randomForest, pROC, gbm, rpart, rpart.plot, glmnet

## Project Workflow
1. Loaded and explored the dataset.
2. Checked dataset dimensions, missing values and duplicates.
3. Removed non-predictive columns such as EmployeeNumber and Over18.
4. Reviewed descriptive statistics and top correlations among numeric variables.
5. Converted categorical variables to factors for modeling.
6. Created exploratory visualizations for attrition trends across employee characteristics.
7. Split the dataset into training and test sets.
8. Built and evaluated Random Forest, Gradient Boosting, CART and Logistic Regression models.
9. Compared models using AUC, Accuracy, Kappa, MAE, RMSE and R-squared.
10. Created model comparison plots for final evaluation.

## Exploratory Analysis Highlights
The exploratory analysis focused on attrition trends across:
- Monthly Income
- Age
- Total Working Years
- Job Role
- OverTime
- Distance From Home
- Years at Company

These visualizations were used to understand turnover patterns before model building.

## Models Used
- Random Forest
- Gradient Boosting Machine (GBM)
- CART
- Logistic Regression

## Evaluation Metrics
The project compares model performance using:
- AUC
- Accuracy
- Kappa
- MAE
- RMSE
- R-squared

It also includes:
- Confusion matrices
- ROC curves
- Variable importance review for Random Forest
- Model comparison charts

## Repository Structure
```text
employee-attrition-prediction/
│
├── README.md
├── .gitignore
├── employee_attrition_prediction.R
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── images/
│   ├── attrition_by_age.png
│   ├── attrition_by_monthly_income.png
│   ├── attrition_by_total_working_years.png
│   ├── attrition_by_job_role.png
│   ├── attrition_by_overtime.png
│   ├── attrition_by_distance_from_home.png
│   ├── attrition_by_years_at_company.png
│   ├── logistic_regression_roc.png
│   ├── random_forest_roc.png
│   ├── gradient_boosting_roc.png
│   ├── cart_roc.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── gradient_boosting_confusion_matrix.png
│   ├── cart_confusion_matrix.png
│   ├── model_comparison_auc_accuracy_kappa.png
│   ├── model_comparison_mae_r_squared_rmse.png
│   └── decision_tree.png
│
└── report/
    └── employee_attrition_project_report.pdf
```

## How to Run
1. Open the project in R or RStudio.
2. Install the required packages if they are not already installed.
3. Keep the CSV file in the project root folder.
4. Run the script from top to bottom.

## Output
The project produces:
- exploratory attrition visualizations
- model summaries
- variable importance output
- confusion matrix plots
- ROC curves
- model comparison plots

## Author
Nitish Harinkhere