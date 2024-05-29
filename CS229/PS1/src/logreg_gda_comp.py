# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:47:34 2024

Compare logreg and GDA performance

@author: Ryan Tsai
"""

import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg

from linear_model import LinearModel

import p01b_logreg
import p01e_gda

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 1 # data set index, 1 or 2
    transform = 1 # if 1, then take natural log of x2
    
    train_path = '../data/ds' + str(dsidx) + '_train.csv'
    eval_path = '../data/ds' + str(dsidx) + '_valid.csv'
    
    """ Logreg training """
    
    # add_intercept=True means adding x_0=1 as the first column
    x,y = util.load_dataset(train_path,add_intercept=True)
    if transform:
        x[:,-1] = np.log(x[:,-1])
    
    logreg_clf = p01b_logreg.LogisticRegression()
    logreg_clf.verbose = False
    logreg_clf.max_iter = 400000; logreg_clf.fit(x,y,solver='newton')
   
    """ Logreg testing """
    
    x,y = util.load_dataset(eval_path,add_intercept=True)
    if transform:
        x[:,-1] = np.log(x[:,-1])
    
    h = logreg_clf.predict(x)
    h = np.reshape(h,h.size)

    num_errs = sum(abs(h-y))
    print('Number of errors in the logreg prediction: ' + str(num_errs))
   
    # Post-processing: calculate decision boundary (where theta.T * x is zero)
    theta0 = logreg_clf.theta[0,0]
    theta1 = logreg_clf.theta[1,0]
    theta2 = logreg_clf.theta[2,0]
    x1 = np.linspace(min(x[:,1]),max(x[:,1]),num=100)
    x2 = (-theta0 - theta1*x1)/theta2
    
    # Plot eval data
    idx_ones = np.asarray(y == 1).nonzero()[0]
    idx_zeros = np.asarray(y == 0).nonzero()[0]
    plt.figure()
    plt.scatter(x[idx_ones,1],x[idx_ones,2],marker="^",s=200,label="y=1")
    plt.scatter(x[idx_zeros,1],x[idx_zeros,2],marker="o",s=50,label="y=0")

    plt.plot(x1,x2,'--k',linewidth=3,label='Logreg decision boundary')
    
    """ GDA training """
    
    x,y = util.load_dataset(train_path,add_intercept=False)
    if transform:
        x[:,-1] = np.log(x[:,-1])
    
    gda_clf = p01e_gda.GDA()
    gda_clf.fit(x,y)
    
    """ GDA testing """
    
    x,y = util.load_dataset(eval_path,add_intercept=False)
    if transform:
        x[:,-1] = np.log(x[:,-1])
        
    h = gda_clf.predict(x)
    h = np.reshape(h,h.size)
    
    num_errs = sum(abs(h-y))
    print('Number of errors in the GDA prediction: ' + str(num_errs))
    
    # Post-processing: calculate decision boundary (where theta.T * x is zero)
    theta0 = gda_clf.theta[0,0]
    theta1 = gda_clf.theta[1,0]
    theta2 = gda_clf.theta[2,0]
    x1 = np.linspace(min(x[:,0]),max(x[:,0]),num=100)
    x2 = (-theta0 - theta1*x1)/theta2
    
    plt.plot(x1,x2,':k',linewidth=3,label='GDA decision boundary')
    
    plt.title("Logreg vs. GDA",{'fontsize':40})
    plt.xlabel("x_1",{'fontsize':30})
    plt.ylabel("x_2",{'fontsize':30})
    plt.legend(loc="upper left",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()