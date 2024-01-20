import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the TF-IDF matrices
X_train = load_npz('features/train_feature_data.npz')
train_labels = np.loadtxt('data/train_labels.txt', dtype=int)  # Corrected the path for training labels

X_val = load_npz('features/val_feature_data.npz')
val_labels = np.loadtxt('data/val_labels.txt', dtype=int)  # Corrected the path for validation labels

# Initialize the Support Vector Classifier
svc = SVC()  # Use class_weight='balanced' argument for balanced class weights

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

# Generate the confusion matrix
conf_matrix: np.ndarray = confusion_matrix(val_labels, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
