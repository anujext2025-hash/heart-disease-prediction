import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import legend
from pandas.conftest import ascending
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)

# fix ssl issue for mac
ssl._create_default_https_context = ssl._create_unverified_context

# load dataset
import pandas as pd

url = 'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv'
df = pd.read_csv(url)
print(df.head())
print("Shape :", df.shape)
print(df.info())

print("Missing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df['target'].value_counts())

num_cols = df.select_dtypes(['number']).columns
print(num_cols)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 -Q1
    lower_limit = Q1 - 1.5*IQR
    upper_limit = Q3 - 1.5*IQR
print("\nlower_limit:", lower_limit,
      "\nupper_limit:", upper_limit,
      "\nIQR:", IQR)
outliers = [(df[col] < lower_limit) | (df[col] > upper_limit)]
print(f"Outliers count: {len(outliers)} outliers")

#outliers through plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sns.boxplot(data=df, y='trestbps', ax=axes[0])
axes[0].set_title("Blood Pressure")

sns.boxplot(data=df, y='chol', ax=axes[1])
axes[1].set_title("Cholestrol")

sns.boxplot(data=df, y='thalach' , ax=axes[2])
axes[2].set_title("Max Heart Rate")
plt.tight_layout()
plt.show()

print(f"Count above 400: {len(df[df['chol'] > 400])}")
print(f"Count above 500: {len(df[df['chol'] > 500])}")

print(df[df['chol'] > 400][['age', 'target', 'chol', 'sex']])
#67 year old female with heart disease has a cholesterol of 564- might be an error

df['chol'] = df['chol'].clip(upper=400)
print(f"Max cholesterol after capping: {df['chol'].max()}")
print(f"Outliers above 400: {len(df[df['chol'] > 400])}")

print("\nGender with disease rate:")
print(df.groupby('sex')['target'].mean())
#females are less(more than half) as compared to male- 75% is based on very few samples
print("\nChest pain type with disease rate:")
print(df.groupby('cp')['target'].mean())
print("\nDisease rate based on age group:")
df['age_group'] = pd.cut(df['age'],
                   bins=[20, 40, 50, 60, 80],
                   labels=['20-40', '40-50', '50-60', '60+'])
print(df.groupby('age_group')['target'].mean())

print(df.groupby('target')['chol'].agg(['mean', 'median', 'max', 'min']))
df['chol_group'] = pd.cut(df['chol'],
                          bins=[0, 200, 240, 280, 400],
                          labels=['Low', 'Normal', 'High', 'Very High'])
print(df.groupby('chol_group')['target'].mean())

plt.figure(figsize=(12,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=0.5,
            )
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

X = df.drop(columns=['target'])
y = df['target']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

#logisticregression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:,1]

#randomforest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

print("=" * 40)
print("Logistic Regression")
print("=" * 40)
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1-score:", f1_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))

print("\n" + "=" * 40)
print("Random Forest")
print("=" * 40)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1-score:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_pred))

#cross_validation
lr_cv = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("Logistic Regression CV scores:", lr_cv)
print(f"Mean: {lr_cv.mean():.4f}(+/- {lr_cv.std():.4f})")

print("Random Forest:", rf_cv)
print(f"Mean: {rf_cv.mean():.4f}(+\- {rf_cv.std():.4f})")

#confusion matrix comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d',
            cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease']
)
axes[0].set_title('Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d',
            cmap='Blues', ax=axes[1],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease']
            )
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.show()
print('Logistic Regression')
print(cm_lr)
print('Random Forest')
print(cm_rf)

feat_imp = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(feat_imp)

plt.figure(figsize=(12, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index,
            palette='coolwarm', hue=feat_imp.index)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()