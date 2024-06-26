import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load data
data = pd.read_excel('data.xlsx', sheet_name='data')
train_data = pd.read_excel('data.xlsx', sheet_name='oversample.train')
test_data = pd.read_excel('data.xlsx', sheet_name='test')

# Prepare and standardize data
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(train_data[['amount', 'second', 'days']])
y_train_svm = train_data['fraud']

# Train SVM model
svm_model = SVC(kernel='linear', C=1.0, gamma='scale')
svm_model.fit(X_train_svm, y_train_svm)

# Save the scaler and model to disk
dump(scaler, 'svm_scaler.joblib')
dump(svm_model, 'svm_model.joblib')
