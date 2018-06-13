import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.datasets import make_hastie_10_2

'''Semi-supervised ssclustering'''
def ssclustering(X_l, y_l, X_u):
    X0 = []
    X1 = []
    for index, i in enumerate(X_l):
        if y_l[index] == 0:
            X0.append(i)
        else:
            X1.append(i)
    X0 = np.array(X0)
    X1 = np.array(X1)
    mean0 = np.mean(X0, axis=0)
    mean1 = np.mean(X1, axis=0)
    while(True):
        C0 = X0
        C1 = X1
        for index, i in enumerate(X_u):
            d0 = np.linalg.norm(i - mean0)
            d1 = np.linalg.norm(i - mean1)
            if d0 < d1:
                C0 = np.append(C0, [i], axis=0)
            else:
                C1 = np.append(C1, [i], axis=0)
        mean0n = np.mean(C0, axis=0)
        mean1n = np.mean(C1, axis=0)
        if (mean0n == mean0).all() and (mean1n == mean1).all():
            break
        else:
            mean0 = mean0n
            mean1 = mean1n
    X = np.append(C0, C1, axis=0)
    y = [0 for x in range(0, C0.shape[0])]
    y.extend([1 for xx in range(0, C1.shape[0])])
    y = np.array(y)
    return X, y



''' Check out data '''
#magic = pd.read_csv('magic04data.csv', header=None)
#print(magic.head());
#magic.describe()
X, y = make_classification(n_samples=3000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)



'''Get data in a format fitting sklearn'''
#magic[10] = pd.Categorical(magic[10])
#magic[10] = magic[10].cat.codes
#magic.tail()
# Get data as arrays, shuffle, and separate features from labels
#X_raw = magic.values
#np.random.shuffle(X_raw)
#y = X_raw[:,-1]  #label
#X = X_raw[:,:-1]  #data


''' Normalize X to get unit standard deviation '''
col_std = np.std(X, axis=1)
for j in range(X.shape[1]): #shape[1] returns the num of columns
    X[:,j] = X[:,j] / col_std[j]





''' Select data for supervised and unsupervised training '''
#In order to change X and the corresponding y at the same time
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #np.random.permutation() 随机排列
    return a[p], b[p]

num_labeled = 25
num_unlabeled = 18995#[0, 10, 20, 40, 80, 160, 320, 640]
num_unlabeledPlot = [0, 10, 20, 40, 80, 160, 320, 640]
n = 20

X, y = unison_shuffled_copies(X, y)

averageError = [];
averageError2 = [];
#averageError3 = [];
averageError4 = [];

error = []
for i in range(n):    #run 20 times to calculate 20 error rates
        
        '''train a model supervised to see how it works '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #randomly separate training data and test data

        X_l = X_train[:num_labeled]
        y_l = y_train[:num_labeled]
        #X_u = X[num_unlabeled]
        #y_u = y[num_unlabeled]
        X_u = X_train[num_labeled:num_labeled + num_unlabeled]
        y_u = y_train[num_labeled:num_labeled + num_unlabeled]

        '''Supervised Learning'''
        # Train on labeled data
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_l, y_l)

        # Predict labels for unlabeled data
        y_u_pred = clf.predict(X_u)
        accuracy = accuracy_score(y_pred=y_u_pred, y_true=y_u)
        error.append(1-accuracy)

averageError.append(np.mean(error))
        
for index,i in enumerate(num_unlabeledPlot):
    error2 = []
    error4 = []
    for i in range(n):    #run 40 times to calculate 40 error rates

        '''Semi-supervised _ Self Learning'''
        
        count = 0
        num_labeled1 = 25
        y_l = y_train[:25]
        X_l = X_train[:25]
        for i in range(num_unlabeledPlot[index]/2):
                X_u = X_train[num_labeled1:num_labeled1+2]
                y_u = y_train[num_labeled1:num_labeled1+2]

                clf = LinearDiscriminantAnalysis()
                clf.fit(X_l, y_l)

                y_u_pred = clf.predict(X_u)
                proba = clf.predict_proba(X_u)
                if abs(proba[0][0] - proba[0][1]) >0.9:
                   y_l = np.append(y_l, y_u_pred[0])
                   X_l = np.concatenate((X_l, X_u[0].reshape(1,2)))
                if abs(proba[1][0] - proba[1][1]) >0.9:
                   y_l = np.append(y_l, y_u_pred[1])
                   X_l = np.concatenate((X_l, X_u[1].reshape(1,2)))
                num_labeled1 += 2
                
                
        X_u1 = X_train[num_labeled1:]
        y_u1 = y_train[num_labeled1:]
        y_u_pred1 = clf.predict(X_u1)
        accuracy1 = accuracy_score(y_pred=y_u_pred1, y_true=y_u1)
        error2.append(1-accuracy1)
            

        ###### SS-Clustering ######
        if num_unlabeledPlot[index] == 0:
            X_trnall = X_train[: 25]
            y_trnall = y_train[: 25]
        else:
            X_trnall, y_trnall = ssclustering(X_l, y_l, X_u)
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)   
        train_predictions = clf_lda.predict(X_test)
        error4.append(1 - accuracy_score(y_test, train_predictions))
        
    averageError2.append(np.mean(error2))


    #averageError3.append(np.mean(error3))

    averageError4.append(np.mean(error4))


averageError1 = [averageError, averageError, averageError, averageError, averageError, averageError, averageError, averageError]

plt.figure()
line_1,=plt.plot(num_unlabeledPlot, averageError1, 'r-',label='LDA Supervised Learning')
#line_2,=plt.plot(num_unlabeledPlot, averageError3, 'g-',label='Semi-supervised Self Learning with Prior Probabilities')
line_3,=plt.plot(num_unlabeledPlot, averageError2, 'y-',label='Semi-supervised Self Learning')
line_4,=plt.plot(num_unlabeledPlot, averageError4, 'b-',label='Semi-supervised Clustering')
plt.xlabel('Numbers of Unlabeled Samples')
plt.ylabel('Error Rate')
plt.legend([line_1, line_3, line_4], ['LDA Supervised Learning','Semi-supervised Self Learning', 'Semi-supervised Clustering'])
plt.show()


          
