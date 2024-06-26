import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load data
data = pd.read_excel('data.xlsx', sheet_name='data')
train_data = pd.read_excel('data.xlsx', sheet_name='oversample.train')
test_data = pd.read_excel('data.xlsx', sheet_name='test')

# Prepare and standardize data
scaler = StandardScaler()
X_train_ksvm = pd.DataFrame(scaler.fit_transform(train_data[['amount', 'second', 'days']]), columns=['amount', 'second', 'days'])
y_train_ksvm = train_data['fraud']

# Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_train_ksvm)
X_train_ksvm['cluster'] = kmeans.labels_

# Train SVM on clusters
cluster_svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, gamma='scale'))
cluster_svm_model.fit(X_train_ksvm, y_train_ksvm)

# Save the scaler, kmeans, and cluster SVM model to disk
dump(scaler, 'kmeans_scaler.joblib')
dump(kmeans, 'kmeans_model.joblib')
dump(cluster_svm_model, 'cluster_svm_model.joblib')
