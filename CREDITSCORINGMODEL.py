# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# Load the German Credit dataset from UCI Repository
# Dataset link: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount', 
    'savings_account', 'employment', 'installment_rate', 'personal_status', 
    'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans', 
    'housing', 'existing_credits', 'job', 'liable_people', 'telephone', 
    'foreign_worker', 'default'
]
df = pd.read_csv(url, sep=' ', header=None, names=column_names)
# Convert target variable: 1 = Good (no default), 2 = Bad (default)
df['default'] = df['default'].apply(lambda x: 0 if x == 1 else 1)
# Feature Engineering
df['credit_amount_to_income'] = df['credit_amount'] / (df['duration'] * 100)  # Proxy for monthly income
df['age_squared'] = df['age'] ** 2
df['credit_amount_log'] = np.log(df['credit_amount'])
df['installment_rate_squared'] = df['installment_rate'] ** 2
df['employment_stability'] = np.where(df['employment'].isin(['A71', 'A72']), 0, 1)  # Unemployed or <1 year = 0, else = 1
# Define features and target
X = df.drop('default', axis=1)
y = df['default']
# Split data into train and test sets
#75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
# Preprocessing pipeline
numeric_features = ['duration', 'credit_amount', 'installment_rate', 'residence_since', 
                   'age', 'existing_credits', 'liable_people', 'credit_amount_to_income',
                   'age_squared', 'credit_amount_log', 'installment_rate_squared']
categorical_features = ['checking_account', 'credit_history', 'purpose', 'savings_account', 
                        'employment', 'personal_status', 'other_debtors', 'property', 
                        'other_installment_plans', 'housing', 'job', 'telephone', 
                        'foreign_worker', 'employment_stability']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
# Model training and evaluation function
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
    # Train model
    pipeline.fit(X_train, y_train)
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    # Print metrics
    print(f"Model: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    return pipeline, roc_auc
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}
# Train and evaluate models
results = {}
for name, model in models.items():
    print("\n" + "="*50)
    print(f"Training {name}...")
    trained_model, roc_auc = train_evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'model': trained_model, 'roc_auc': roc_auc}
# Compare model performance
print("\n" + "="*50)
print("Model Comparison (ROC-AUC):")
for name, result in results.items():
    print(f"{name}: {result['roc_auc']:.4f}")
# Select best model
best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} with ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
# Feature importance for Random Forest (if it's the best model)
if best_model_name == "Random Forest":
    # Get feature names after preprocessing
    feature_names = numeric_features + list(
        best_model.named_steps['preprocessor'].named_transformers_['cat']
        .named_steps['onehot'].get_feature_names_out(categorical_features)
    )
    # Get feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()