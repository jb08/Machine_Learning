import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
#%matplotlib inline

import pickle
import sklearn
from sklearn import svm, metrics # this is an example of using SVM

n_features = 784

def preprocess(images):
    #this function is suggested to help build your classifier. 
    #You might want to do something with the images before 
    #handing them to the classifier. Right now it does nothing.

    return [i.flatten() for i in images]

def build_classifier(images, labels):
    #this will actually build the classifier. In general, it
    #will call something from sklearn to build it, and it must
    #return the output of sklearn. Right now it does nothing.
    
    best_metric = 10
    gamma_input = (1.0/n_features) * best_metric

    #print "gamma = " + str(gamma_input)

    classifier = svm.SVC(gamma = gamma_input, kernel='rbf', degree= 2)

    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set_1.p', 'w'))
    pickle.dump(training_labels, open('training_labels_1.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

if __name__ == "__main__":

    # Code for loading data
    clean_images, labels = load_mnist(digits=range(0, 10), path='.')
    
    # preprocessing
    images = preprocess(clean_images)
    #print len(images[1])
    
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA

    #for i in range(0,1000):
    #   if i%2 == 0:

    X = 500

    training_set = images[0:X]
    training_labels = labels[0:X]
    testing_set = images[X:2*X]
    testing_labels = labels[X:2*X]

    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    #classifier = build_classifier(training_set, training_labels)
    #save_classifier(classifier, training_set, training_labels)
    classifier = pickle.load(open('classifier_1.p'))
    predicted = classify(testing_set, classifier)

    #print predicted
    #print testing_labels

    print_mislabeled_digits = False

    if(print_mislabeled_digits):

        count = 0;
        for i in range(0,X):
            if count > 5:
                break
            if predicted[i] != testing_labels[i]:
                #print testing_labels[i]
                #print testing_set[i]
                plt.imshow(clean_images[i+X], cmap = 'gray')
                plt.title('Predicted: '+ str(predicted[i]) + '; actually: '+ str(testing_labels[i]))
                plt.show()
                count +=1

    print error_measure(predicted, testing_labels)
    print metrics.confusion_matrix(testing_labels, predicted)