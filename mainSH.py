#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:37:59 2023

@author: Sem Hoogteijling
s.hoogteijling@umcutrecht.nl

This python code is developed for the manuscript'Machine learning for (non-)epileptic
tissue detection from the intraoperative electrocorticogram'
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
import copy
from scipy.stats import beta
from scipy.stats import norm
import delong
from loadData import loadSpectralFeatures

#%% load data

X_train, y_train, X_test, y_test = loadSpectralFeatures() #load spectral features

#%% some functions

def getMetrics(y,y_pred, CI = False, alpha = 0.05, print_ = False):
    TN,FP, FN, TP = confusion_matrix(y,y_pred).ravel()

    n0 = TN+FP
    n1 = TP+FN
    p = n1/(n0+n1)
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    Se = TP/n1
    Sp = TN/n0
    
    ppv = Se*p/(Se*p + (1-Sp)* (1-p))
    npv = (Sp*(1-p))/((1-Se)*p+Sp*(1-p))
    metrics = {'acc': acc, 'Se': Se, 'Sp': Sp, 'ppv': ppv, 'npv': npv}
    
    if CI:
        z = norm.ppf(1 - alpha / 2) # Critical value for standard normal distribution
        
        logPPV = np.log(Se*p/((1-Sp)*(1-p)))
        varlogPPV = ((1-Se)/Se)*(1/n1)+((Sp/(1-Sp))*(1/n0))
        upper_bound_ppv = np.exp(logPPV+z*np.sqrt(varlogPPV))/(1+np.exp(logPPV+z*np.sqrt(varlogPPV)))
        lower_bound_ppv = np.exp(logPPV-z*np.sqrt(varlogPPV))/(1+np.exp(logPPV-z*np.sqrt(varlogPPV))) 
        ppv = [ppv, lower_bound_ppv, upper_bound_ppv]
        
        logNPV = np.log(Sp*(1-p)/((1-Se)*p))
        varlogNPV = ((Se/(1-Se))*(1/n1)) + ((1-Sp)/Sp)*(1/n0)
        
        upper_bound_npv = np.exp(logNPV+z*np.sqrt(varlogNPV))/(1+np.exp(logNPV+z*np.sqrt(varlogNPV)))
        lower_bound_npv = np.exp(logNPV-z*np.sqrt(varlogNPV))/(1+np.exp(logNPV-z*np.sqrt(varlogNPV)))  
        npv = [npv,lower_bound_npv,upper_bound_npv]
        
        acc = [acc, beta.ppf(alpha / 2, TN+TP, FP+FN + 1), beta.ppf(1 - alpha / 2, TN+TP + 1, FP+FN)]
        Se = [Se, beta.ppf(alpha / 2, TP, FN + 1), beta.ppf(1 - alpha / 2, TP + 1, FN)]
        Sp = [Sp, beta.ppf(alpha / 2, TN, FP + 1), beta.ppf(1 - alpha / 2, TN + 1, FP)]
        
        metrics = {'acc': acc, 'Se': Se, 'Sp': Sp, 'ppv': ppv, 'npv': npv}
        
        if print_:
            for metric in metrics.keys():
                print(metric+':', np.round(metrics[metric][0]*100,1), '(' + str(np.round(metrics[metric][1]*100,1))+ '-'+ str(np.round(metrics[metric][2]*100,1))+')%')
        

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
summary_metrics_train = pd.DataFrame(columns = ['acc', 'Se', 'Sp', 'ppv','npv','auc']) #create summary metrics dataframe
summary_metrics_val = pd.DataFrame(columns = ['acc', 'Se', 'Sp', 'ppv','npv','auc'])


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
    if metrics_val['Sp']>spec_best:
        spec_best = metrics_val['Sp']
        ETC_best = copy.deepcopy(ETC)
        scaler_best = copy.deepcopy(scaler)
        thr95_best = copy.deepcopy(thr95)
        fpr_train_best = copy.deepcopy(fpr_train)
        tpr_train_best = copy.deepcopy(tpr_train)


#print results
print('-'*65, '\n SUMMARY METRICS 5FOLD CV:')
print('      Training   | Validation \n      Mean (SD)  | Mean (SD)')
print('Acc : %0.2f (%0.2f)' %(summary_metrics_train['acc'].mean(),summary_metrics_train['acc'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['acc'].mean(),summary_metrics_val['acc'].std()))
print('Sens: %0.2f (%0.2f)' %(summary_metrics_train['Se'].mean(),summary_metrics_train['Se'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['Se'].mean(),summary_metrics_val['Se'].std()))
print('Spec: %0.2f (%0.2f)' %(summary_metrics_train['Sp'].mean(),summary_metrics_train['Sp'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['Sp'].mean(),summary_metrics_val['Sp'].std()))
print('PPV : %0.2f (%0.2f)' %(summary_metrics_train['ppv'].mean(),summary_metrics_train['ppv'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['ppv'].mean(),summary_metrics_val['ppv'].std()))
print('NPV : %0.2f (%0.2f)' %(summary_metrics_train['npv'].mean(),summary_metrics_train['npv'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['npv'].mean(),summary_metrics_val['npv'].std()))
print('AUC : %0.2f (%0.2f)' %(summary_metrics_train['auc'].mean(),summary_metrics_train['auc'].std()) + '| %0.2f (%0.2f)' %(summary_metrics_val['auc'].mean(),summary_metrics_val['auc'].std()))

## Calculate performance on the test set
#scale data
X_test_scaled = scaler_best.transform(X_test)

#obtain prediction ETC
y_test_pred = ETC_best.predict_proba(X_test_scaled)[:,1] 
y_test_pred95 = (y_test_pred >= thr95).astype(bool)

#print results
print('-'*65, '\n METRICS TEST SET:')
metrics_test = getMetrics(y_test,y_test_pred95, CI = True, print_=True)

#AUC sklearn
fpr_test, tpr_test, thresholds_test = roc_curve(y_test,y_test_pred)
print('AUC:', auc(fpr_test, tpr_test))

#AUC deLong
alpha = 0.95
auc_delong, auc_cov = delong.delong_roc_variance(y_test,y_test_pred)
auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
ci = norm.ppf(lower_upper_q,loc=auc_delong,scale=auc_std)
ci[ci > 1] = 1
print('AUC deLong:', np.round(auc_delong,2), '('+str(np.round(ci[0],2))+'-'+ str(np.round(ci[1],2))+')')


#compute performance metrics ETC on train subsets
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



#%% SHAP analysis
explainer = shap.Explainer(ETC_best)
shap_test = explainer(X_test_scaled)

#plot summary plot
shap.summary_plot(shap_test[:,:,1], plot_type='dot', plot_size = (7,6), axis_color='#000000',show = False, sort = False, title = True)
