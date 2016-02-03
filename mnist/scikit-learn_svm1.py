"""
================================
Recognizing hand-written digits 
================================

An example showing how the scikit-learn SVM classifier can be used to recognize images of hand-written digits given kaggle MNIST curated dataset.

This code is a modification to the one developed by Gael Varoquaux <gael dot varoquaux at normalesup dot org>
and available at http://scikit-learn.org/stable/_downloads/plot_digits_classification.py

"""

print(__doc__)

# Author: Arun Radhakrishnan

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# Import numpy for array manipulation and the genfromtxt method
import numpy as np

# This is the class that mimics the scikit datasets
class KaggleDataset(object):
    data =  np.array([1, 2, 3])
    target =  np.array([1, 2, 3])

# Load Kaggle dataset from training csv file
def load_kaggle_data():

   # Use path where the training file is available
   # data = np.genfromtxt('C:/Users/arun/Desktop/kaggle/train_sample.csv',delimiter=',')
   data = np.genfromtxt('train.csv',delimiter=',')

   # Remove first row as its label name
   data = np.delete(data, 0, 0)

   # First column is the target values
   target = data[:,0]

   # Remove the first column and what remains is the sample data
   data = np.delete(data, 0, 1)
   kaggle_data = KaggleDataset()
   kaggle_data.data =  data
   kaggle_data.target = target
   print(kaggle_data.data)
   print(kaggle_data.target)
   return kaggle_data

# The digits_kaggle dataset
digits_kaggle = load_kaggle_data()

# The digits dataset
digits = digits_kaggle


# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.target)

data = digits.data

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

# What's the confusion matrix: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
