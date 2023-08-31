#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:37:59 2023

@author: Sem Hoogteijling
s.hoogteijling@umcutrecht.nl

This python code is developed for the manuscript'Explainable AI reveals epilespy
and non-epilepsy specific spectral band powers in the intra-operative electrocorticography'
by Hoogteijling et al.

"""
#%%
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import shap
import seaborn as sns
import copy

from loadData import loadSpectralFeatures

sns.set_theme(style = 'ticks', font = 'Times New Roman') #set theme for plots
#%% load data
X_train, y_train, X_test, y_test = loadSpectralFeatures() #load spectral features

#%% some functions

def getMetrics(y,y_pred):
    tn,fp, fn, tp = confusion_matrix(y,y_pred).ravel()
    metrics = {
        'acc': (tn+tp)/(tp+tn+fp+fn),
        'sens': tp/(tp+fn),
        'spec': tn/(tn+fp),
        'ppv': tp/(tp+fp),
        'npv': tn/(fn+tn),
        }
    return metrics
    
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 25, fill = '█', printEnd = "\r"):
    total = len(iterable)
    #progress printing function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    #initial Call
    printProgressBar(0)
    #update progress bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    #print new line when complete
    print()

#%% 5 fold cross validation and test set evaluation

spec_best = 0
summary_metrics_train = pd.DataFrame(columns = ['acc', 'sens', 'spec', 'ppv','npv','auc']) #create summary metrics dataframe
summary_metrics_val = pd.DataFrame(columns = ['acc', 'sens', 'spec', 'ppv','npv','auc'])


for i in progressBar(range(5), prefix = 'Progress 5 fold CV:', suffix = 'Complete'):
    
    #create new ETC model
    ETC = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.45, 
                          min_samples_leaf=100, min_samples_split=70, 
                          n_estimators=100, class_weight='balanced')
    
    #select training and validation data for this fold
    X_train_fold = X_train[X_train['5foldCV'] != i].drop(columns = ['5foldCV'])    
    y_train_fold = y_train[X_train['5foldCV'] != i]
    X_val_fold = X_train[X_train['5foldCV'] == i].drop(columns = ['5foldCV'])  
    y_val_fold = y_train[X_train['5foldCV'] == i]
    
    #scale data
    scaler = StandardScaler().set_output(transform = 'pandas')
    X_train_fold = scaler.fit_transform(X_train_fold[X_train_fold.columns])
    X_val_fold = scaler.transform(X_val_fold) #scale validation subset based on training subsets
    
    #train model
    ETC.fit(X_train_fold,y_train_fold)
    
    #obtain predictions ETC
    y_train_fold_pred = ETC.predict_proba(X_train_fold)[:,1]
    y_val_fold_pred = ETC.predict_proba(X_val_fold)[:,1]
    
    #calculate classification threshold based on 95% spec in train set
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_fold,y_train_fold_pred)
    spec_train = 0.95 
    idx95 = (np.abs(fpr_train-(1-spec_train))).argmin() # find index with 95% specificity
    thr95 = thresholds_train[idx95]
    
    #compute performance metrics ETC on train and validation subsets
    y_train_fold_pred95 = (y_train_fold_pred >= thr95).astype(bool)
    y_val_fold_pred95 = (y_val_fold_pred >= thr95).astype(bool)

    metrics_train = getMetrics(y_train_fold,y_train_fold_pred95)
    metrics_val = getMetrics(y_val_fold,y_val_fold_pred95)

    #calculate auc's
    fpr_val, tpr_val, thresholds_val = roc_curve(y_val_fold,y_val_fold_pred)
    
    metrics_train['auc'] = auc(fpr_train, tpr_train)
    metrics_val['auc'] = auc(fpr_val, tpr_val)
    
    #save metrics in summary dataframe
    summary_metrics_train.loc[i] = metrics_train
    summary_metrics_val.loc[i] = metrics_val

    #check if this is the best fold yet and if so, save ETC model, scaler, thr95, fpr_train and tpr_train
    if metrics_val['spec']>spec_best:
        spec_best = metrics_val['spec']
        ETC_best = copy.deepcopy(ETC)
        scaler_best = copy.deepcopy(scaler)
        thr95_best = copy.deepcopy(thr95)
        fpr_train_best = copy.deepcopy(fpr_train)
        tpr_train_best = copy.deepcopy(tpr_train)


## Calculate performance on the test set
#scale data
X_test_scaled = scaler_best.transform(X_test)

#obtain prediction ETC
y_test_pred = ETC_best.predict_proba(X_test_scaled)[:,1] 
y_test_pred95 = (y_test_pred >= thr95).astype(bool)

metrics_test = getMetrics(y_test,y_test_pred95)

#compute performance metrics ETC on train and validation subsets
fpr_test, tpr_test, thresholds_test = roc_curve(y_test,y_test_pred)
metrics_test['auc'] = auc(fpr_test, tpr_test)

#plot ROC
plt.figure(figsize = (10,10))
lw = 4

plt.plot(fpr_train_best,tpr_train_best,':',color="k",lw=lw,
    label="Train set (AUC = %0.2f)" % auc(fpr_train_best, tpr_train_best))

plt.plot(fpr_test,tpr_test,color="k",lw=lw,
    label="Test set (AUC = %0.2f)" % auc(fpr_test, tpr_test))    

#edit plot
plt.plot([0, 1], [0, 1], color="k", lw=1, linestyle="--")
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.xlabel("False positive rate", fontsize=30)
plt.ylabel("True positive rate", fontsize=30)
plt.title('Receiver operating characteristic', fontsize=30)
plt.legend(loc="lower right", fontsize=27)
plt.show()


#print results
print('-'*65, '\n SUMMARY METRICS 5FOLD CV:')
print('      Training   | Validation \n      Mean (SD)  | Mean (SD)')
print('Acc : %0.2f (%0.2f)' %(summary_metrics_train['acc'].mean(),summary_metrics_train['acc'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['acc'].mean(),summary_metrics_val['acc'].std()))
print('Sens: %0.2f (%0.2f)' %(summary_metrics_train['sens'].mean(),summary_metrics_train['sens'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['sens'].mean(),summary_metrics_val['sens'].std()))
print('Spec: %0.2f (%0.2f)' %(summary_metrics_train['spec'].mean(),summary_metrics_train['spec'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['spec'].mean(),summary_metrics_val['spec'].std()))
print('PPV : %0.2f (%0.2f)' %(summary_metrics_train['ppv'].mean(),summary_metrics_train['ppv'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['ppv'].mean(),summary_metrics_val['ppv'].std()))
print('NPV : %0.2f (%0.2f)' %(summary_metrics_train['npv'].mean(),summary_metrics_train['npv'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['npv'].mean(),summary_metrics_val['npv'].std()))
print('AUC : %0.2f (%0.2f)' %(summary_metrics_train['auc'].mean(),summary_metrics_train['auc'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['auc'].mean(),summary_metrics_val['auc'].std()))

print('-'*65, '\n METRICS TEST SET:')
print('Acc : %0.2f' %metrics_test['acc'])
print('Sens: %0.2f' %metrics_test['sens'])
print('Spec: %0.2f' %metrics_test['spec'])
print('PPV : %0.2f' %metrics_test['ppv'])
print('NPV : %0.2f' %metrics_test['npv'])
print('AUC : %0.2f' %metrics_test['auc'])


#%% SHAP analysis
explainer = shap.Explainer(ETC_best)
shap_test = explainer(X_test_scaled)

#plot summary plot
shap.summary_plot(shap_test[:,:,1], plot_type='dot', plot_size = (7,6), axis_color='#000000',show = False, sort = False, title = True)
