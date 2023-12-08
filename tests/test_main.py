# import unittest
# from main import hello_world
#
#
# class MainTest(unittest.TestCase):
#     def test_hello(self):
#         self.assertEqual(hello_world(), "Hello, World!")
#
#
# if __name__ == '__main__':
#     unittest.main()
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import LineTokenizer, TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


mapping = open('data/mapping.txt', encoding="utf-8").read()
test_labels = open('data/test_labels.txt', encoding="utf-8").read()
train_labels = open('data/train_labels.txt', encoding="utf-8").read()
val_labels = open('data/val_labels.txt', encoding="utf-8").read()
val_text = open('data/val_text.txt', encoding="utf-8").read()
train_text = open('data/train_text.txt', encoding="utf-8").read()
test_text = open('data/test_text.txt', encoding="utf-8").read()
print(mapping)
lt = LineTokenizer()
tt = TweetTokenizer()
vectorizer = TfidfVectorizer()

# data_length = len(val_labels_split) + len(train_labels_split)
# classes = np.arange(20)
# counts = np.array([])
# percentages = np.array([])
# total_percentage = 0
# for i in range(20):
#     counts = np.append(counts, val_labels_split.count(str(i)) + train_labels_split.count(str(i)))
#     percentages = np.append(percentages, 100 * (val_labels_split.count(str(i)) + train_labels_split.count(str(i))) / data_length)

#split the data into datapoints by separating them by line
tokentext = lt.tokenize(train_text)
tokenlabels = lt.tokenize(train_labels)
print(len(tokentext))
print(len(tokenlabels))

# for i in range(10):
#     print(tokentext[len(tokentext)-i-1] + "\n")
#     print(tokenlabels[len(tokenlabels)-i-1] + "\n")

# tokenize the individual datapoints
for line, i in zip(tokentext, range(len(tokentext))):
    tokentext[i] = tt.tokenize(tokentext[i])

#remove tokens that are only space characters

# X here is the document-term matrix

#applying PCA on this?
#https://stackoverflow.com/questions/59162130/how-can-i-use-the-pca-for-a-term-document-matrix-in-python
X = vectorizer.fit_transform(tokentext)
# print(vectorizer.get_feature_names_out())
# print(X.shape)
# print(X[0][0])
# print(tokentext[0:6])

# print(classes)
# print(counts)
# print(percentages)
# plt.xticks(classes)
# plt.bar(classes, counts)
# plt.xlabel("Class")
# plt.ylabel("Occurrence of class")
# plt.show()
