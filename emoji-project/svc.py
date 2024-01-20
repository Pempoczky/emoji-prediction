import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the TF-IDF matrices
X_train = load_npz('features/train_feature_data.npz')
train_labels = np.loadtxt('data/train_labels.txt', dtype=int)

X_val = load_npz('features/val_feature_data.npz')
val_labels = np.loadtxt('data/val_labels.txt', dtype=int)

# Define the parameter grid for SVC
param_grid = {
    'C': [0,1, 1, 10, 15, 20],  # Regularization parameter
    'kernel': ['poly', 'sigmoid', 'rbf'],  # Kernel type
    'class_weight': ['balanced']  # Class weights
}

# Split the data into training and validation sets for cross-validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, train_labels, test_size=0.2, random_state=42
)

# Initialize the Support Vector Classifier
svc = SVC()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2)

# Fit the grid search to the data
# grid_search.fit(X_train_split, y_train_split)

# Get the best parameters and the best model
# best_params = grid_search.best_params_
# Alternatively, use precomputed parameters below
best_params = {'C': 1, 'kernel': 'rbf', 'class_weight': 'balanced'}
best_svc = SVC(**best_params)
best_svc.fit(X_train, train_labels)

# Print the best parameters
print("\nBest Hyperparameters:")
print(best_params)

# Make predictions on the validation set using the best model
y_pred = best_svc.predict(X_val)

# Evaluate the best classifier
report = classification_report(val_labels, y_pred)
print(f"Classification Report for Best Classifier:\n{report}")

# Save the best model
joblib.dump(best_svc, 'models/best_svc_model.joblib')

# Generate the confusion matrix
conf_matrix = confusion_matrix(val_labels, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
