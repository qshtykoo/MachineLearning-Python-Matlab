import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

from sklearn.cluster import KMeans
import csv

#try different classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

Y_data = pd.read_csv('test.csv', header=None)
Y_raw = Y_data.values

magic = pd.read_csv('train.csv', header=None)
X_raw = magic.values
X_data = X_raw[:,1:]
X_label = X_raw[:,0]


#initialize centers
def initialize_center(X_data):
    cluster_1 = X_data[0:100,:]
    cluster_2 = X_data[100:200,:]
    center_1 = np.average(cluster_1, axis=0)
    center_2 = np.average(cluster_2, axis=0)
    return center_1, center_2

#normalizing features (each column)
def normalize_feature(X_data):
    averages=np.average(X_data,axis=0)
    #calculate standard deviation
    X_data-=averages
    X2=np.square(X_data)
    var=np.average(X2,axis=0)
    sd=np.sqrt(var)
    #perform z-score transformation
    X_data/=sd
    return X_data




pca = PCA(n_components = 148)
#use PCA to fit train.csv
X_data = pca.fit_transform(X_data)

#get the two centers of train.csv
[center_1, center_2] = initialize_center(X_data)
centers = np.zeros((2,len(center_1)))
centers[0,:] = center_1
centers[1,:] = center_2
#normalize train.csv
X_data = normalize_feature(X_data)

#apply the transform fitted on train.csv onto unlabelled test.csv
Y_raw = pca.transform(Y_raw)
#initialize k-means with two centers calculated from the train.csv
kmeans = KMeans(n_clusters=2, init = centers)
#cluster on unlabelled test.csv
kmeans.fit(Y_raw)
#normalize test.csv
Y_raw = normalize_feature(Y_raw)
clustered_labels = kmeans.labels_ + 1


error = [];
train_error = [];
#leave-one-out cross validation on train.csv
kf = KFold(n_splits=200)
for train_index, test_index in kf.split(X_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = X_label[train_index], X_label[test_index]

    #X_train = np.concatenate((X_train, Y_raw), axis=0)
    #y_train = np.concatenate((y_train, clustered_labels), axis=0)
    #clf = SVC(kernel='sigmoid')
    #clf.fit(X_train, y_train)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    #clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         #algorithm="SAMME",
                         #n_estimators=200)
    #clf.fit(X_train, y_train)

    #clf = KNeighborsClassifier(n_neighbors = 30)
    #clf.fit(X_train, y_train)
    
    predicted_labels = clf.predict(X_test)
    predicted_labels_train = clf.predict(X_train)

    train_error.append(np.sum(abs(predicted_labels_train - y_train))/len(y_train))
    error.append(np.sum(abs(predicted_labels - y_test))/len(y_test))

average_test_error = np.mean(error)
average_train_error = np.mean(train_error)
print(average_test_error)
print(average_train_error)

#test_labels_final = clf.predict(Y_raw)
#np.savetxt("test_labels_LDA.csv", test_labels_final, delimiter=",")



#use test.csv as training data for linear SVC and apply the classifier onto
#the train.csv

#clfSVM = SVC(kernel='rbf')
#clfSVM.fit(Y_raw, clustered_labels)
#p_clustered_labels = clfSVM.predict(X_data)

#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                        #n_estimators=200)
#bdt.fit(Y_raw, clustered_labels)
#p_clustered_labels = bdt.predict(X_data)

#neigh = KNeighborsClassifier(n_neighbors = 30)
#neigh.fit(Y_raw, clustered_labels)
#p_clustered_labels = neigh.predict(X_data)

#clfNew = LinearDiscriminantAnalysis()
#clfNew.fit(Y_raw, clustered_labels)
#p_clustered_labels = clfNew.predict(X_data)


#error_c_s = np.sum(abs(p_clustered_labels - X_label))/len(X_label)
#print(error_c_s)
    
#predicted_test_labels1 = clfSVM.predict(Y_raw)
#predicted_test_labels2 = clf.predict(Y_raw)
#print(np.sum(abs(predicted_test_labels1 - predicted_test_labels2)))

#with open('test_labels.csv', 'w') as csvFile:
    #writer = csv.writer(csvFile)
    #for label in predicted_test_labels:
        #writer.writerows(label)

#csvFile.close()


