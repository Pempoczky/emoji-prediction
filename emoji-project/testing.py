import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# If I import the .npz I get an error where 1 out of over 60000 features is missing
# Therefore I do the preprocessing steps in the testing script itself

vectorizer = joblib.load('models/vectorizer_model.joblib')
test_text = open('data/test_text.txt', encoding="utf-8").read()
lineSeparatedTestText = test_text.splitlines()
X_test = vectorizer.transform(lineSeparatedTestText)

# Load the TF-IDF test matrix (So this one has an issue as described above)
# X_test = load_npz('features/test_feature_data.npz')
test_labels = np.loadtxt('data/test_labels.txt', dtype=int)

# Load a pre-trained model
model = joblib.load('models/best_svc_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the ensemble classifier
report = classification_report(test_labels, y_pred)
print(f"Classification Report:\n{report}")

# Generate the confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
