import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

'''Semi-supervised Self Learning_co-training'''
def co_training(X_l, y_l, X_u):
    X_labeled = X_l
    y_labeled = y_l
    X_unlabeled = X_u

    while X_unlabeled.shape[0] != 0:
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_labeled,y_labeled)
        train_pre_lda = clf_lda.predict(X_unlabeled)

        clf_D = DecisionTreeClassifier()
        clf_D.fit(X_labeled,y_labeled)
        train_pre_KNN = clf_D.predict(X_unlabeled)

        newX_l=[]
        newy_1=[]
        dele=[]

        for i in range(0,len(X_unlabeled)):
            if train_pre_lda[i] == train_pre_KNN[i]:
                newX_l.append(X_unlabeled[i])
                newy_1.append(train_pre_lda[i])
                dele.append(i)
        newX_l=np.array(newX_l)
        newy_l=np.array(newy_1)
        if newX_l.shape[0] == 0:
            break
        X_labeled = np.append(X_labeled,newX_l,axis=0)
        y_labeled = np.append(y_labeled,newy_1)
        X_unlabeled = np.delete(X_unlabeled, dele,axis=0)
    return X_labeled,y_labeled

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
magic = pd.read_csv('magic04data.csv', header=None)
#print(magic.head());
#magic.describe()


'''Get data in a format fitting sklearn'''
magic[10] = pd.Categorical(magic[10])
magic[10] = magic[10].cat.codes
magic.tail()
# Get data as arrays, shuffle, and separate features from labels
X_raw = magic.values
np.random.shuffle(X_raw)
y = X_raw[:,-1]  #label
X = X_raw[:,:-1]  #data


''' Normalize X to get unit standard deviation '''
col_std = np.std(X, axis=1)
for j in range(X.shape[1]): #shape[1] returns the num of columns
    X[:,j] = X[:,j] / col_std[j]


'''train a model supervised to see how it works 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  #randomly separate training data and test data

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
#print(accuracy)
'''

''' Select data for supervised and unsupervised training '''
#In order to change X and the corresponding y at the same time
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a)) #np.random.permutation() 随机排列
    return a[p], b[p]

num_labeled = 25
num_unlabeled = [0, 10, 20, 40, 80, 160, 320, 640]

X, y = unison_shuffled_copies(X, y) #let (X,y) randomly arranged

err_lda = {}
err_co = {}
err_sl = {}
err_clu = {}
log_sl = {}
log_lda = {}
log_co = {}
log_clu = {}
literation = range(0,40)
for n in literation: #run 40 times
    #get train and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y) #split the train and test data

    e_lda = []
    e_co = []
    e_clu = []
    e_sl = []
    l_lda = []
    l_co = []
    l_clu = []
    l_sl = []
    num_labeled1 = 25 #specifically for self-learning 
    y_l = y[:25]
    X_l = X[:25]

    for index,i in enumerate(num_unlabeled):
        X_labeled = X_train[:num_labeled]
        y_labeled = y_train[:num_labeled]
        X_unlabeled = X_train[num_labeled:num_labeled + num_unlabeled[index]]
        y_unlabeled = y_train[num_labeled:num_labeled + num_unlabeled[index]]

        #### supervised-lDA ####
        X_trainset = X_train[:num_labeled]
        y_trainset = y_train[:num_labeled]
        # Train on labeled data
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trainset, y_trainset)
        # Predict labels for unlabeled data
        train_pre_lda = clf_lda.predict(X_test)
        accuracy_lda = accuracy_score(y_pred=train_pre_lda, y_true=y_test)
        e_lda.append(1-accuracy_lda)
        l_lda.append(log_loss(y_pred=train_pre_lda, y_true=y_test))

        ###### SS-Clustering ######
        if num_unlabeled[index] == 0:
            X_trnall = X_train[: num_labeled]
            y_trnall = y_train[: num_labeled]
        else:
            X_trnall, y_trnall = ssclustering(X_labeled, y_labeled, X_unlabeled)
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trnall, y_trnall)
        train_predictions = clf_lda.predict(X_test)
        e_clu.append(1 - accuracy_score(y_test, train_predictions))
        l_clu.append(log_loss(y_test, train_predictions))

        #### co-training ####
        X_trainset, y_trainset = co_training(X_labeled, y_labeled, X_unlabeled)
        # Train on labeled data
        clf_lda = LinearDiscriminantAnalysis()
        clf_lda.fit(X_trainset, y_trainset)
        # Predict labels for unlabeled data
        train_pre_co = clf_lda.predict(X_test)
        accuracy_co = accuracy_score(y_pred=train_pre_co, y_true=y_test)
        e_co.append(1 - accuracy_co)
        l_co.append(log_loss(y_pred=train_pre_co, y_true=y_test))

        #### self-learning ####
        for j in range(num_unlabeled[index]/2):
            X_u = X[num_labeled1:num_labeled1+2]
            y_u = y[num_labeled1:num_labeled1+2]

            clf_sl = LinearDiscriminantAnalysis()
            clf_sl.fit(X_l, y_l)

            train_pre_sl = clf_sl.predict(X_u)
            proba = clf_sl.predict_proba(X_u)
            if abs(proba[0][0] - proba[0][1]) > 0.9:
                y_l = np.append(y_l, train_pre_sl[0])
                X_l = np.concatenate((X_l, X_u[0].reshape(1, 10)))
            if abs(proba[1][0] - proba[1][1]) > 0.9:
                y_l = np.append(y_l, train_pre_sl[1])
                X_l = np.concatenate((X_l, X_u[1].reshape(1, 10)))
            num_labeled1 += 2

        X_l1 = X[:25+num_unlabeled[index]]
        y_l1 = y[:25+num_unlabeled[index]]
        X_u1 = X[25+num_unlabeled[index]:]
        y_u1 = y[25+num_unlabeled[index]:]
        clf_sl = LinearDiscriminantAnalysis()
        clf_sl.fit(X_l1,y_l1)
        train_pre_sl1 = clf_sl.predict(X_u1)
        accuracy_sl = accuracy_score(y_pred=train_pre_sl1, y_true=y_u1)
        e_sl.append(1-accuracy_sl)
        l_sl.append(log_loss(y_pred=train_pre_sl1, y_true=y_u1))

    err_lda[n] = e_lda
    err_co[n] = e_co
    err_sl[n] = e_sl
    err_clu[n] = e_clu
    log_sl[n] = l_sl
    log_lda[n] = l_lda
    log_co[n] = l_co
    log_clu[n] = l_clu

avererr_lda = []
avererr_co = []
avererr_sl = []
avererr_clu = []
averlog_clu = []
averlog_sl = []
averlog_lda = []
averlog_co = []
# to get the mean of 80 times running result
for index,i in enumerate(num_unlabeled):
    sum_lda = 0
    sum_co = 0
    sum_sl = 0
    sum_clu = 0
    sum_log_clu = 0
    sum_log_sl = 0
    sum_log_lda = 0
    sum_log_co = 0
    for n in err_lda.keys():
        sum_lda = sum_lda + err_lda[n][index]
        sum_co = sum_co + err_co[n][index]
        sum_sl = sum_sl + err_sl[n][index]
        sum_clu = sum_clu + err_clu[n][index]
        sum_log_clu = sum_log_clu + log_clu[n][index]
        sum_log_sl = sum_log_sl + log_sl[n][index]
        sum_log_lda = sum_log_lda + log_lda[n][index]
        sum_log_co = sum_log_co + log_co[n][index]
    # for each index for num_unlabeled [0, 10, 20, 40, 80, 160, 320, 640]
    # get the mean of 100 times running result
    avererr_lda.append(sum_lda / len(literation))
    avererr_co.append(sum_co / len(literation))
    avererr_sl.append(sum_sl / len(literation))
    avererr_clu.append(sum_clu / len(literation))
    averlog_clu.append(sum_log_clu / len(literation))
    averlog_sl.append(sum_log_sl / len(literation))
    averlog_lda.append(sum_log_lda / len(literation))
    averlog_co.append(sum_log_co / len(literation))

plt.figure()
line_1,=plt.plot(num_unlabeled, avererr_lda, 'r-',label='LDA Supervised Learning')
line_2,=plt.plot(num_unlabeled, avererr_co, 'g-',label='Semi-supervised Co-training')
line_3,=plt.plot(num_unlabeled, avererr_sl, 'y-',label='Semi-supervised Self Learning')
line_4,=plt.plot(num_unlabeled, avererr_clu, 'b-',label='Semi-supervised Clustering')
plt.xlabel('Numbers of Unlabeled Samples')
plt.ylabel('Error Rate')
plt.legend([line_1, line_2, line_3, line_4], ['LDA Supervised Learning', 'Semi-supervised Co-training', 'Semi-supervised Self Learning', 'Semi-supervised Clustering'])
plt.show()

plt.figure()
line_1,=plt.plot(num_unlabeled, averlog_lda,'r-', label='LDA Supervised Learning')
line_2,=plt.plot(num_unlabeled, averlog_co, 'g-',label='Semi-supervised Co-training')
line_3,=plt.plot(num_unlabeled, averlog_sl, 'y-',label='Semi-supervised Self Learning')
line_4,=plt.plot(num_unlabeled, averlog_clu, 'b-',label='Semi-supervised Clustering')
plt.xlabel('Numbers of Unlabeled Samples')
plt.ylabel('Log-likelihood')
#plt.legend([line_1, line_2, line_3], ['LDA Supervised Learning', 'Semi-supervised Co-training', 'Semi-supervised Self Learning'])
plt.legend([line_1, line_2, line_3, line_4], ['LDA Supervised Learning', 'Semi-supervised Co-training', 'Semi-supervised Self Learning', 'Semi-supervised Clustering'])
plt.show()












