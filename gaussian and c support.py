import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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


X = df.drop("PacemakerImplantation", axis= 1)
y = df['PacemakerImplantation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 10 
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test_selected)

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train_resampled)

y_pred_gnb = gnb.predict(X_test_scaled)
svc = SVC(random_state=42)
svc.fit(X_train_scaled, y_train_resampled)
y_pred_svc = svc.predict(X_test_scaled)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

print("Accuracy (Gaussian Naive Bayes): {:.2f}%".format(accuracy_gnb * 100))
print("Accuracy (SVC): {:.2f}%".format(accuracy_svc * 100))