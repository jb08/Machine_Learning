import pickle
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier                        

def preprocess(images):
    return [i.flatten() for i in images]

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

def classify(images, classifier):
    return classifier.predict(images)

def boosting_A(training_set, training_labels, testing_set, testing_labels):
	#build classifier
	classifier = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 50)
	classifier.fit(training_set, training_labels)
	#Prediction
	predicted = classify(testing_set,classifier)
	print error_measure(predicted, testing_labels)
	confusion_matrix = metrics.confusion_matrix(testing_labels, predicted)
	print confusion_matrix
	return confusion_matrix
	
def boosting_B(training_set, training_labels, testing_set, testing_labels):
	#build classifier
	base_classifier = svm.SVC(gamma = 10.0/784)
	#base_classifier.fit(training_set, training_labels)
	classifier = AdaBoostClassifier(base_estimator=base_classifier, algorithm='SAMME', n_estimators = 50)
	classifier.fit(training_set, training_labels)
	#Prediction
	predicted = classify(testing_set,classifier)
	print error_measure(predicted, testing_labels)
	confusion_matrix = metrics.confusion_matrix(testing_labels, predicted)
	print confusion_matrix
	return confusion_matrix






if __name__ == "__main__":
	# Code for loading data
    clean_images, labels = load_mnist(digits=range(0,10),path='.')	
    num_images = len(labels)    

    # preprocessing
    images = preprocess(clean_images)
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK SET DIFFERENT OF DATA
    
    X = 1000

    training_set = images[:X] 
    training_labels = labels[:X] 
    testing_set = images[X:2*X]
    testing_labels = labels[X:2*X]

    boosting_A(training_set, training_labels, testing_set, testing_labels)
    #print "Boosting B:"
    #boosting_B(training_set, training_labels, testing_set, testing_labels)

