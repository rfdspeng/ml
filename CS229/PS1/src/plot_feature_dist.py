# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:22:03 2024

Plot joint distribution of input features

@author: Ryan Tsai
"""

import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg

from linear_model import LinearModel

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 1 # data set index, 1 or 2
    
    train_path = '../data/ds' + str(dsidx) + '_train.csv'
    eval_path = '../data/ds' + str(dsidx) + '_valid.csv'
    
    x,y = util.load_dataset(train_path,add_intercept=False)
    
    nbins = 30
    
    plt.figure()
    plt.hist2d(x[:,0],x[:,1],bins=nbins)
    
    plt.figure()
    plt.hist(x[:,0],bins=nbins)
    plt.title('x1')
    
    plt.figure()
    plt.hist(x[:,1],bins=nbins)
    plt.title('x2')
    
    plt.figure()
    plt.hist(np.log(x[:,1]),bins=nbins)
    plt.title('ln(x2)')