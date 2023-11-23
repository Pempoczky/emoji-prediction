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


mapping = open('data/mapping.txt', encoding="utf-8").read()
test_labels = open('data/test_labels.txt', encoding="utf-8").read()
train_labels = open('data/train_labels.txt', encoding="utf-8").read()
val_labels = open('data/val_labels.txt', encoding="utf-8").read()
val_text = open('data/val_text.txt', encoding="utf-8").read()
train_text = open('data/train_text.txt', encoding="utf-8").read()
test_text = open('data/test_text.txt', encoding="utf-8").read()
print(mapping)
classes = np.arange(20)
counts = np.array([])
for i in range(20):
    counts = np.append(counts, val_labels.count(str(i)) + train_labels.count(str(i)))
print(classes)
print(counts)
plt.xticks(classes)
plt.bar(classes, counts)
plt.xlabel("Class")
plt.ylabel("Occurrence of class")
plt.show()
