
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pydantic import BaseModel
from sklearn.linear_model import SGDClassifier
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

working_directory = os.getcwd()
print(working_directory)
path = 'total_tavr_v1.xlsx'
df = pd.read_excel(path)

df['DeltaPR'] = df['DeltaPR'].replace("#N/A", np.nan)
df['DeltaPR'] = pd.to_numeric(df['DeltaPR'])
df['DeltaPR'] = df['DeltaPR'].fillna(df['DeltaPR'].mean())
df['DeltaQRS'] = df['DeltaQRS'].replace("#N/A", np.nan)
df['DeltaQRS'] = pd.to_numeric(df['DeltaQRS'])
df['DeltaQRS'] = df['DeltaQRS'].fillna(df['DeltaQRS'].mean())
df['FirstdegreeAVblock'] = df['FirstdegreeAVblock'].fillna(df['FirstdegreeAVblock'].mean())
df['DiastolicBP'] = df['DiastolicBP'].fillna(df['DiastolicBP'].mean())
df['SystolicBP'] = df['SystolicBP'].fillna(df['SystolicBP'].mean())
df['LVEF'] = df['LVEF'].fillna(df['LVEF'].mean())
df['PR'] = df['PR'].replace("#N/A", np.nan)
df['PR'] = pd.to_numeric(df['PR'])
df['PR'] = df['PR'].fillna(df['PR'].mean())
df['LVOT'] = df['LVOT'].fillna(df['LVOT'].mean())
df['BSA'] = df['BSA'].fillna(df['BSA'].mean())


null_values = df.isnull().sum()

# Display the count of null values for each column
print(null_values)
X = df.drop("PacemakerImplantation", axis= 1)
y = df['PacemakerImplantation']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

# Feature selection
k_best = SelectKBest(score_func=f_classif, k=14)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Perform SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_selected, y_train)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_oversampled)
X_test_scaled = scaler.transform(X_test_selected)

# SVM + SGD Classifier
svm_sgd = SGDClassifier(loss='hinge', random_state=42)
svm_param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'penalty': ['l1', 'l2']}
svm_grid_search = GridSearchCV(svm_sgd, svm_param_grid, cv=5)
svm_grid_search.fit(X_train_scaled, y_train_oversampled)
svm_best_model = svm_grid_search.best_estimator_
svm_predictions = svm_best_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': [3, 5, 7, 9]}
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5)
knn_grid_search.fit(X_train_scaled, y_train_oversampled)
knn_best_model = knn_grid_search.best_estimator_
knn_predictions = knn_best_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print('-----------------------------')
# Print the accuracies
print("SVM + SGD Accuracy:", svm_accuracy)

f1 = f1_score(y_test, svm_predictions)
print('F1 score:', f1)

brier_score = brier_score_loss(y_test, svm_predictions)
print('Brier score:', brier_score)
print('-----------------------------')
print("KNN Accuracy:", knn_accuracy)


f1 = f1_score(y_test, knn_predictions)
print('F1 score:', f1)

brier_score = brier_score_loss(y_test, knn_predictions)
print('Brier score:', brier_score)
print('-----------------------------')