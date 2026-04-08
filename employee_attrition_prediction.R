# =========================================================
# Employee Attrition Prediction
# Author: Nitish Harinkhere
# Description: Predictive analytics project using R to
# analyze employee attrition and compare multiple models.
# Dataset: IBM HR Analytics Employee Attrition dataset
# =========================================================


# =========================================================
# 1. Package installation and library loading
# =========================================================

# Run once if packages are not already installed
# install.packages(c("readr", "dplyr", "psych", "ggplot2", "caret", "randomForest", "pROC", "gbm", "rpart", "rpart.plot", "glmnet"))

# Load required libraries
library(readr)
library(dplyr)
library(psych)
library(ggplot2)
library(caret)
library(randomForest)
library(pROC)
library(gbm)
library(rpart)
library(rpart.plot)
library(glmnet)


# =========================================================
# 2. Load the dataset
# =========================================================

# Load employee attrition dataset from the project directory
df <- read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# =========================================================
# 3. Initial data exploration and quality checks
# =========================================================

# Review dataset dimensions, missing values, and duplicate records
dim(df)
sum(is.na(df))
any(duplicated(df))


# =========================================================
# 4. Descriptive statistics
# =========================================================

# Generate descriptive statistics and summary information
describe(df)
summary(df)


# =========================================================
# 5. Data preprocessing
# =========================================================

# Remove identifier and non-informative columns that do not add predictive value
df <- df[, !names(df) %in% c("EmployeeNumber", "Over18")]

# Separate numeric and character variables for correlation analysis
numerical_df <- df[, sapply(df, is.numeric)]
character_df <- df[, sapply(df, is.character)]

# Convert categorical variables to factors for modeling
df <- df %>% mutate(across(where(is.character), as.factor))


# =========================================================
# 6. Correlation analysis
# =========================================================

# Calculate correlation matrix for numeric variables and extract top correlations
correlation_matrix <- cor(numerical_df)
upper_triangle <- correlation_matrix[upper.tri(correlation_matrix, diag = FALSE)]
top_indices <- order(upper_triangle, decreasing = TRUE)[1:10]

top_correlations <- data.frame(
  Var1 = colnames(correlation_matrix)[row(correlation_matrix)[upper.tri(correlation_matrix)]][top_indices],
  Var2 = colnames(correlation_matrix)[col(correlation_matrix)[upper.tri(correlation_matrix)]][top_indices],
  Correlation = upper_triangle[top_indices]
)

print(top_correlations)


# =========================================================
# 7. Exploratory data visualization
# =========================================================

# Create exploratory visualizations to examine attrition patterns across key employee attributes

# 1. Attrition Trend by Monthly Income
ggplot(df, aes(x = MonthlyIncome, fill = Attrition)) +
  geom_density(alpha = 0.5) +
  labs(title = "Attrition Trend by Monthly Income",
       x = "Monthly Income", y = "Density") +
  theme_minimal()

# 2. Attrition Trend by Age
ggplot(df, aes(x = Age, fill = Attrition)) +
  geom_histogram(position = "dodge", binwidth = 1) +
  labs(title = "Attrition Trend by Age",
       x = "Age", y = "Count") +
  theme_minimal()

# 3. Attrition Trend by Total Working Years
ggplot(df, aes(x = TotalWorkingYears, fill = Attrition)) +
  geom_histogram(position = "dodge", binwidth = 1) +
  labs(title = "Attrition Trend by Total Working Years",
       x = "Total Working Years", y = "Count") +
  theme_minimal()

# 4. Attrition Trend by Job Role
ggplot(df, aes(x = JobRole, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Trend by Job Role",
       x = "Job Role", y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5. Attrition Trend by OverTime
ggplot(df, aes(x = OverTime, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Trend by OverTime",
       x = "OverTime", y = "Proportion") +
  theme_minimal()

# 6. Attrition Trend by Distance From Home
ggplot(df, aes(x = DistanceFromHome, fill = Attrition)) +
  geom_bar(position = "fill") +
  labs(title = "Attrition Trend by Distance From Home",
       x = "Distance From Home", y = "Proportion") +
  theme_minimal()

# 7. Attrition Trend by Years at Company
ggplot(df, aes(x = YearsAtCompany, fill = Attrition)) +
  geom_histogram(position = "dodge", binwidth = 1) +
  labs(title = "Attrition Trend by Years at Company",
       x = "Years at Company", y = "Count") +
  theme_minimal()


# =========================================================
# 8. Train-test split
# =========================================================

# Partition the dataset into training and test sets
set.seed(123)
train_index <- createDataPartition(df$Attrition, p = 0.8, list = FALSE)
train_df <- df[train_index, ]
test_df <- df[-train_index, ]


# =========================================================
# 9. Random Forest model
# =========================================================

# Train Random Forest model
rf_model <- randomForest(Attrition ~ ., data = train_df, importance = TRUE, ntree = 500)

# Print Random Forest model summary
print(rf_model)

# Extract and display variable importance
rf_importance <- randomForest::importance(rf_model)
print(rf_importance)
varImpPlot(rf_model)

# Generate predictions on the test set
rf_predictions <- predict(rf_model, newdata = test_df)

# Evaluate Random Forest performance using confusion matrix
rf_conf_matrix <- confusionMatrix(rf_predictions, test_df$Attrition)
print(rf_conf_matrix)

# Plot confusion matrix for Random Forest
rf_conf_matrix_df <- as.data.frame(rf_conf_matrix$table)
ggplot(rf_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix (Random Forest)", x = "Predicted", y = "Actual") +
  theme_minimal()

# Calculate Random Forest evaluation metrics
rf_accuracy <- rf_conf_matrix$overall['Accuracy']
rf_kappa <- rf_conf_matrix$overall['Kappa']
rf_prob_predictions <- predict(rf_model, newdata = test_df, type = "prob")[,2]
rf_roc_curve <- roc(test_df$Attrition, rf_prob_predictions)
rf_auc_value <- auc(rf_roc_curve)
print(paste("Random Forest AUC: ", rf_auc_value))

# Plot ROC curve for Random Forest
plot(rf_roc_curve, main = "ROC Curve for Random Forest Model", col = "blue")
abline(a = 0, b = 1, lty = 2, col = "red")

# Extract top 10 important features from Random Forest
important_features <- rownames(rf_importance)[order(rf_importance[, "MeanDecreaseGini"], decreasing = TRUE)][1:10]
print(important_features)


# =========================================================
# 10. Convert target variable for binary numeric modeling
# =========================================================

# Convert Attrition to numeric format for models and metrics that require binary numeric input
train_df$Attrition <- ifelse(train_df$Attrition == "Yes", 1, 0)
test_df$Attrition <- ifelse(test_df$Attrition == "Yes", 1, 0)


# =========================================================
# 11. Gradient Boosting Machine (GBM) model
# =========================================================

# Train Gradient Boosting Machine model
gbm_model <- gbm(
  formula = Attrition ~ .,
  data = train_df,
  distribution = "bernoulli",
  n.trees = 500,
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  n.minobsinnode = 10
)

# Generate probability predictions on the test set
gbm_prob_predictions <- predict(gbm_model, newdata = test_df, n.trees = gbm_model$n.trees, type = "response")
gbm_predictions <- ifelse(gbm_prob_predictions > 0.5, 1, 0)

# Evaluate Gradient Boosting model performance
gbm_conf_matrix <- confusionMatrix(factor(gbm_predictions, levels = c(0, 1)), factor(test_df$Attrition, levels = c(0, 1)))
print("Gradient Boosting Model Evaluation:")
print(gbm_conf_matrix)

gbm_accuracy <- gbm_conf_matrix$overall['Accuracy']
gbm_kappa <- gbm_conf_matrix$overall['Kappa']
gbm_roc_curve <- roc(test_df$Attrition, gbm_prob_predictions)
gbm_auc_value <- auc(gbm_roc_curve)
print(paste("Gradient Boosting AUC: ", gbm_auc_value))

# Plot confusion matrix for Gradient Boosting
gbm_conf_matrix_df <- as.data.frame(gbm_conf_matrix$table)
ggplot(gbm_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "pink") +
  labs(title = "Confusion Matrix (Gradient Boosting)", x = "Predicted", y = "Actual") +
  theme_minimal()

# Plot ROC curve for Gradient Boosting
plot(gbm_roc_curve, main = "ROC Curve (Gradient Boosting)", col = "green", lwd = 2)


# =========================================================
# 12. CART model
# =========================================================

# Train CART classification model
cart_model <- rpart(
  Attrition ~ .,                     # Response (dependent) variable
  data = train_df,                   # Training data
  method = "class",                  # Classification method
  control = rpart.control(cp = 0.01) # Complexity parameter
)

# Plot CART decision tree
rpart.plot(cart_model)

# Generate predictions on the test set
cart_predictions <- predict(cart_model, newdata = test_df, type = "class")

# Evaluate CART model performance
cart_conf_matrix <- confusionMatrix(factor(cart_predictions, levels = c(0, 1)), factor(test_df$Attrition, levels = c(0, 1)))
print("CART Model Evaluation:")
print(cart_conf_matrix)

cart_accuracy <- cart_conf_matrix$overall['Accuracy']
cart_kappa <- cart_conf_matrix$overall['Kappa']
cart_prob_predictions <- predict(cart_model, newdata = test_df, type = "prob")[, 2]
cart_roc_curve <- roc(test_df$Attrition, cart_prob_predictions)
cart_auc_value <- auc(cart_roc_curve)
print(paste("CART AUC: ", cart_auc_value))

# Plot confusion matrix for CART
cart_conf_matrix_df <- as.data.frame(cart_conf_matrix$table)
ggplot(cart_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "skyblue") +
  labs(title = "Confusion Matrix (CART)", x = "Predicted", y = "Actual") +
  theme_minimal()

# Plot ROC curve for CART
plot(cart_roc_curve, main = "ROC Curve (CART)", col = "blue", lwd = 2)


# =========================================================
# 13. Logistic Regression model
# =========================================================

# Train Logistic Regression model
logit_model <- glm(Attrition ~ ., data = train_df, family = binomial)

# Generate probability predictions on the test set
logit_prob_predictions <- predict(logit_model, newdata = test_df, type = "response")
logit_predictions <- ifelse(logit_prob_predictions > 0.5, 1, 0)

# Evaluate Logistic Regression model performance
logit_conf_matrix <- confusionMatrix(factor(logit_predictions, levels = c(0, 1)), factor(test_df$Attrition, levels = c(0, 1)))
print("Logistic Regression Model Evaluation:")
print(logit_conf_matrix)

logit_accuracy <- logit_conf_matrix$overall['Accuracy']
logit_kappa <- logit_conf_matrix$overall['Kappa']
logit_roc_curve <- roc(test_df$Attrition, logit_prob_predictions)
logit_auc_value <- auc(logit_roc_curve)
print(paste("Logistic Regression AUC: ", logit_auc_value))

# Plot confusion matrix for Logistic Regression
logit_conf_matrix_df <- as.data.frame(logit_conf_matrix$table)
ggplot(logit_conf_matrix_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(title = "Confusion Matrix (Logistic Regression)", x = "Predicted", y = "Actual") +
  theme_minimal()

# Plot ROC curve for Logistic Regression
plot(logit_roc_curve, main = "ROC Curve (Logistic Regression)", col = "purple", lwd = 2)


# =========================================================
# 14. Model comparison metrics
# =========================================================

# Collect AUC values across all models
auc_values <- c(
  "Random Forest" = rf_auc_value,
  "Gradient Boosting" = gbm_auc_value,
  "CART" = cart_auc_value,
  "Logistic Regression" = logit_auc_value
)

# Collect Accuracy values across all models
accuracy_values <- c(
  "Random Forest" = rf_accuracy,
  "Gradient Boosting" = gbm_accuracy,
  "CART" = cart_accuracy,
  "Logistic Regression" = logit_accuracy
)

# Collect Kappa values across all models
kappa_values <- c(
  "Random Forest" = rf_kappa,
  "Gradient Boosting" = gbm_kappa,
  "CART" = cart_kappa,
  "Logistic Regression" = logit_kappa
)

# Print AUC, Accuracy, and Kappa values
print("Model Comparison - AUC Values:")
print(auc_values)
print("Model Comparison - Accuracy Values:")
print(accuracy_values)
print("Model Comparison - Kappa Values:")
print(kappa_values)


# =========================================================
# 15. Additional classification metric calculations
# =========================================================

# Define a function to calculate MAE, RMSE, and R-squared for classification models
calculate_classification_metrics <- function(predictions, true_values) {
  mae <- mean(abs(predictions - true_values))
  rmse <- sqrt(mean((predictions - true_values)^2))
  r2 <- 1 - (sum((predictions - true_values)^2) / sum((true_values - mean(true_values))^2))
  return(c(MAE = mae, RMSE = rmse, R_squared = r2))
}

# Convert predictions to numeric for MAE, RMSE, and R-squared calculation
rf_predictions_numeric <- as.numeric(rf_predictions)
gbm_predictions_numeric <- as.numeric(gbm_predictions)
cart_predictions_numeric <- as.numeric(cart_predictions)
logit_predictions_numeric <- as.numeric(logit_predictions)

# Calculate metrics for Random Forest
rf_metrics <- calculate_classification_metrics(rf_predictions_numeric, test_df$Attrition)
rf_mae <- rf_metrics['MAE']
rf_rmse <- rf_metrics['RMSE']
rf_r2 <- rf_metrics['R_squared']

# Calculate metrics for Gradient Boosting
gbm_metrics <- calculate_classification_metrics(gbm_predictions_numeric, test_df$Attrition)
gbm_mae <- gbm_metrics['MAE']
gbm_rmse <- gbm_metrics['RMSE']
gbm_r2 <- gbm_metrics['R_squared']

# Calculate metrics for CART
cart_metrics <- calculate_classification_metrics(cart_predictions_numeric, test_df$Attrition)
cart_mae <- cart_metrics['MAE']
cart_rmse <- cart_metrics['RMSE']
cart_r2 <- cart_metrics['R_squared']

# Calculate metrics for Logistic Regression
logit_metrics <- calculate_classification_metrics(logit_predictions_numeric, test_df$Attrition)
logit_mae <- logit_metrics['MAE']
logit_rmse <- logit_metrics['RMSE']
logit_r2 <- logit_metrics['R_squared']

# Collect all MAE, RMSE, and R-squared values
mae_values <- c(
  "Random Forest" = rf_mae,
  "Gradient Boosting" = gbm_mae,
  "CART" = cart_mae,
  "Logistic Regression" = logit_mae
)

rmse_values <- c(
  "Random Forest" = rf_rmse,
  "Gradient Boosting" = gbm_rmse,
  "CART" = cart_rmse,
  "Logistic Regression" = logit_rmse
)

r2_values <- c(
  "Random Forest" = rf_r2,
  "Gradient Boosting" = gbm_r2,
  "CART" = cart_r2,
  "Logistic Regression" = logit_r2
)

# Print MAE, RMSE, and R-squared values
print("Model Comparison - MAE Values:")
print(mae_values)
print("Model Comparison - RMSE Values:")
print(rmse_values)
print("Model Comparison - R-squared Values:")
print(r2_values)


# =========================================================
# 16. Comparison visualizations
# =========================================================

# Create comparison data frames for model evaluation metrics
comparison_df1 <- data.frame(
  Model = rep(names(auc_values), 3),
  Metric = rep(c("AUC", "Accuracy", "Kappa"), each = length(auc_values)),
  Value = c(auc_values, accuracy_values, kappa_values)
)

comparison_df2 <- data.frame(
  Model = rep(names(auc_values), 3),
  Metric = rep(c("MAE", "RMSE", "R-squared"), each = length(auc_values)),
  Value = c(mae_values, rmse_values, r2_values)
)

# Plot comparison for AUC, Accuracy, and Kappa
ggplot(comparison_df1, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Model Comparison (AUC, Accuracy, Kappa)", x = "Model", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot comparison for MAE, R-squared, and RMSE
ggplot(comparison_df2, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Metric, scales = "free_y") +
  labs(title = "Model Comparison (MAE, R-squared, RMSE)", x = "Model", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# =========================================================
# End of employee attrition prediction workflow
# =========================================================