
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib

# Step 2: Load Dataset
df = pd.read_csv('Loan_default.csv')

# Step 3: Explore and Clean the Data
print("First 5 rows of the dataset:")
print(df.head())

# Handle missing values (Imputation)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])))
df[df.select_dtypes(include=[np.number]).columns] = df_imputed

# Handle categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Check for missing values after preprocessing
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Step 4: Split Data into Features and Target
X = df.drop(columns=['Default'])  # Assuming 'Default' is the target variable
y = df['Default']

# Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Step 7: Initialize and Train Logistic Regression Model
print("\nTraining Logistic Regression model...")
logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Step 8: Predict and Evaluate Logistic Regression Model
y_pred_logreg = logreg_model.predict(X_test_scaled)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

print("\nLogistic Regression Confusion Matrix:")
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
print(cm_logreg)

print("\nLogistic Regression ROC AUC Score:")
roc_auc_logreg = roc_auc_score(y_test, logreg_model.predict_proba(X_test_scaled)[:, 1])
print(roc_auc_logreg)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize ROC Curve
fpr, tpr, _ = roc_curve(y_test, logreg_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_logreg)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Visualize Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, logreg_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Step 9: Initialize and Train Random Forest Model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 10: Predict and Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test_scaled)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

print("\nRandom Forest ROC AUC Score:")
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
print(roc_auc_rf)

# Step 11: Hyperparameter Tuning
print("\nPerforming Hyperparameter Tuning using GridSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=skf, scoring='roc_auc')
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Train the best model
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(X_test_scaled)

print("\nBest Random Forest Classification Report (after tuning):")
print(classification_report(y_test, y_pred_best_rf))

print("\nBest Random Forest Confusion Matrix (after tuning):")
cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)
print(cm_best_rf)

print("\nBest Random Forest ROC AUC Score (after tuning):")
roc_auc_best_rf = roc_auc_score(y_test, best_rf_model.predict_proba(X_test_scaled)[:, 1])
print(roc_auc_best_rf)

# Save the final model
joblib.dump(best_rf_model, 'loan_default_model.pkl')
print("\nModel saved as 'loan_default_model.pkl'")
