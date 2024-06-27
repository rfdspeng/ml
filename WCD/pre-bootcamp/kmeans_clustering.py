# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:39:38 2024

@author: Ryan Tsai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':
    plt.close('all')
    
    # Generate data
    np.random.seed(10)
    x1 = np.random.randn(10,2) + np.array([4,4]) # 10 2-D datapoints centered around (4,4)
    x2 = np.random.randn(10,2) + np.array([-4,-4]) # 10 2-D datapoints centered around (-4,-4)
    x = np.vstack((x1,x2))
    plt.scatter(x[:,0],x[:,1])
    
    # Initialize KMeans object
    kmeans = KMeans(n_clusters=2,n_init='auto')
    
    # Clustering
    kmeans.fit(x)
    
    # Centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Plot data after clustering
    plt.figure()
    plt.scatter(x[:,0],x[:,1],c=labels,cmap='viridis')
    plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='X',s=100,label='Final Centroids')
    plt.title('K-Means Clustering using sklearn')
    plt.legend()