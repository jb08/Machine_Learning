import pickle
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics, datasets


def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.
    return [i.flatten() for i in images]

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    classifier = KNN(n_neighbors=3,weights='distance')
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_2.p', 'w'))
    pickle.dump(training_set, open('training_set_2.p', 'w'))
    pickle.dump(training_labels, open('training_labels_2.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

if __name__ == "__main__":

    # Code for loading data
    images, labels = load_mnist(digits=range(0,10),path='.')	
    num_images = len(labels)    

    # preprocessing
    images = preprocess(images)
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK SET DIFFERENT OF DATA
    
    X = 10000

    training_set = images[:X] 
    training_labels = labels[:X] 
    testing_set = images[X:2*X]
    testing_labels = labels[X:2*X]

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    classifier = build_classifier(training_set, training_labels)
    save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_2.p'))
    predicted = classify(testing_set, classifier)
    print error_measure(predicted, testing_labels)
    print metrics.confusion_matrix(testing_labels, predicted)
