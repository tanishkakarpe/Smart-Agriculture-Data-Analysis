import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load your dataset
data = pd.read_csv("C:/Users/DELL/Desktop/Crop_recommendation.csv")

# Adding an ID column to your data (if not already present)
data['ID'] = np.arange(1, len(data) + 1)

# Preprocessing
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])  # Encode the crop names

X = data.drop(['label', 'ID'], axis=1)  # Drop the ID column as it's not needed for training
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and label encoder
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Clustering: KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Save KMeans model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("Model and Label Encoder saved successfully!")
