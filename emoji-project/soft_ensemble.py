import numpy as np
from sklearn.ensemble import VotingClassifier
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

# Load the pre-trained models
svc = joblib.load('models/best_svc_model.joblib')
knn = joblib.load('models/best_knn_classifier_model.joblib')

# Allow for probability estimates for soft voting
svc.probability = True
knn.probability = True

# Create a Voting Classifier
ensemble_classifier = VotingClassifier(
    estimators=[
        ('svc', svc),
        ('knn', knn),
    ],
    voting='soft'  # weighted voting, the decision tree model is excluded because it cannot provide
                   # probability estimates
                   # for the other two classes, the 'soft' argument leads the VotingClassifier to call predict_proba(X)
                   # instead of predict(X)
)

# Fit the Voting Classifier to the training data
ensemble_classifier.fit(X_train, train_labels)

# Save the ensemble
joblib.dump(ensemble_classifier, 'models/soft_ensemble_classifier.joblib')

# Make predictions on the validation set
y_pred = ensemble_classifier.predict(X_val)

# Evaluate the ensemble classifier
report = classification_report(val_labels, y_pred)
print(f"Classification Report for Ensemble Classifier:\n{report}")

# Generate the confusion matrix
conf_matrix = confusion_matrix(val_labels, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
