# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:26:38 2024

@author: Ryan Tsai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

if __name__ == '__main__':
    plt.close('all')
    
    # Generate data
    np.random.seed(10)
    x1 = np.random.randn(10,2) + np.array([4,4]) # 10 2-D datapoints centered around (4,4)
    x2 = np.random.randn(10,2) + np.array([-4,-4]) # 10 2-D datapoints centered around (-4,-4)
    x = np.vstack((x1,x2))
    plt.scatter(x[:,0],x[:,1])
    
    # Perform hierarchical clustering using linkage
    linked = linkage(x,method='single')
    
    # Set maximum threshold
    max_inconsistency = 1.1
    #max_inconsistency = 1.5
    #max_inconsistency = 0.5
    cluster_labels = fcluster(linked,max_inconsistency,criterion='inconsistent')
    
    # Plot dendrogram
    plt.figure()
    dendrogram(linked,p=5,truncate_mode='level',labels=cluster_labels)
    plt.title('Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    plt.axhline(max_inconsistency)
    
    # Plot samples after clustering
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap='viridis')