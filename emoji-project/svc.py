import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import load_npz
import joblib

# Load the TF-IDF matrices
X_train = load_npz('features/train_feature_data.npz')
train_labels = np.loadtxt('data/train_labels.txt', dtype=int)  # Corrected the path for training labels

X_val = load_npz('features/val_feature_data.npz')
val_labels = np.loadtxt('data/val_labels.txt', dtype=int)  # Corrected the path for validation labels
print(X_val)
print(X_train)
# Initialize the Support Vector Classifier
svc = SVC()

# Fit the classifier to the training data
svc.fit(X_train, train_labels)

# Make predictions on the validation set
y_pred = svc.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(val_labels, y_pred)
report = classification_report(val_labels, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the trained classifier
joblib.dump(svc, 'models/svc_model.joblib')
