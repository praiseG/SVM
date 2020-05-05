#!/usr/local/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("tools/")
from email_preprocess import preprocess
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score
from time import time

mdl = svm.SVC(kernel='rbf', gamma='auto', C=10000)
# uncomment if you would like to test training on a smaller training set
# features_train = features_train[:round(len(features_train)/100)] 
# labels_train = labels_train[:round(len(labels_train)/100)] 

# train model
to = time()
mdl.fit(features_train, labels_train)
print("fitting time: ", round(time()-to, 3), "s")

# predict using test dataset
t1 = time()
pred = mdl.predict(features_test)
print("prediction time: ", round(time()-t1, 3), "s")

print(len(pred))
print("1s: ", np.count_nonzero(pred == 1))
print("0s: ", np.count_nonzero(pred == 0))

# calculate model accuracy
acc = accuracy_score(pred, labels_test)
print("accuracy:" , acc)
#########################################################


