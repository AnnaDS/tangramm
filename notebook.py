# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:25:32 2015

@author: annachystiakova
"""

import scipy as sp
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

D=pd.read_csv("/Users/annachystiakova/Documents/tangramm_test.csv")
D.columns=[x.strip() for x in D.columns]

print("Data split.")
Dtrain=D[D['test']==0]
Dtest=D[D['test']==1]
#r=list(D.columns-['test', 'class'])
r=list(D.columns.difference(['test', 'class']))
#Data frame of train samples
Dtrain_x=Dtrain[r]
#Data frame of test samples
Dtest_x=Dtest[r]

scaler = StandardScaler()
y_train=Dtrain['class']
y_test=Dtest['class']
#Feature's names
features=Dtest_x.columns

#Scale of features (standartization)
scaler.fit(Dtrain_xn)  # Don't cheat - fit only on training data
X_train = scaler.transform(Dtrain_x)
X_test = scaler.transform(Dtest_x)  



def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi
    
comp=[x/10 for x in range(1,10)]
k=0
best=pd.DataFrame()
#best=best.append(pd.Series({'model':'Random Forest', 'estimator':gs_rfr.best_estimator_, 'score': gs_rfr.best_score_, 'params':gs_rfr.best_params_}), ignore_index=True)
for k in comp:
    scores=[]
    for i in range(0,len(features)):
        s = calc_MI(list(zip(*X_train))[i], list(train_y), 10)
        scores.append(s)
    
    mii=find_indices(np.array(scores), lambda e: e > k)#comp[k])
    print(len(mii))
    X_train_mi=X_train[:,mii]
    X_test_mi=X_test[:,mii]
    features_mi=features[mii]
    
    print("CV starts.")    
    #Support vector regression
    svr = svm.SVR()   
    param_grid = [{'alpha' : 10.0**-np.arange(1,7),'l1_ratio':[.05, .15, .5, .7, .9, .95, .99, 1]}]
    parameters_svr = {'kernel':('linear', 'rbf'), 'C':list(range(1, 100))}
    gs_svr = GridSearchCV(svr,parameters,n_jobs=8,verbose=1)
    gs_svr.fit(X_train_mi, y_train)
    best=best.append(pd.Series({'model':'SVR', 'k':k,'estimator':gs_svr.best_estimator_, 'score': gs_svr.best_score_, 'params':gs_svr.best_params_}), ignore_index=True)

    #Kernel Ridge regression
    kr=KernelRidge()
    parameters_kr = { 'gamma': [0.01, 0.001, 0.0001], 'kernel': ('linear', 'rbf')}
    gs_kr = GridSearchCV(kr,parameters_kr,n_jobs=8,verbose=1)
    gs_kr.fit(X_train_mi, y_train)
    best=best.append(pd.Series({'model':'KernelRidge', 'k':k,'estimator':gs_kr.best_estimator_, 'score': gs_kr.best_score_, 'params':gs_kr.best_params_}), ignore_index=True)

    #Random Forest regression
    rfr=RandomForestRegressor()
    parameters_rfr = {'n_estimators': [500, 700, 1000, 10000], 'max_depth': [None, 1, 2, 3, 4, 5,6], 'min_samples_split': [1, 2, 3]}
    gs_rfr = GridSearchCV(rfr,parameters_rfr,n_jobs=8,verbose=1)
    gs_rfr.fit(X_train_mi, y_train)
    best=best.append(pd.Series({'model':'RandomForest', 'k':k,'estimator':gs_rfr.best_estimator_, 'score': gs_rfr.best_score_, 'params':gs_rfr.best_params_}), ignore_index=True)


#Select best options fom model selections according to the results using list of models and parameters

V=best.score.tolist().index(max(best.score))
BV=best.ix[[V]]
mii=find_indices(np.array(scores), lambda e: e > float(BV.k))

X_train_mi=X_train[:,mii]
X_test_mi=X_test[:,mii]
features_mi=features[mii]
s = BV.estimator.tolist()[0]#svm.SVR(C=99, kernel='linear')
s.fit(X_train_mi, y_train)
    
print("Selected model with params:")
print(str(BV.estimator.tolist()[0]))
print("Best score on the train data:")
print(s.score(X_train_mi, y_train))
print("Best score on test data:")
print(s.score(X_test_mi, y_test.values))
print("MSE of prediction:")
print(np.average(abs(s.predict(X_test_mi)-y_test.values)**2))
print("Average relative accuracy:")
print(100-np.average(abs((s.predict(X_test_mi)-y_test.values)/y_test.values)))