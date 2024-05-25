import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg

from linear_model import LinearModel


def main(train_path,eval_path,pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    
    # Training
    x,y = util.load_dataset(train_path,add_intercept=False)
    clf = GDA()
    clf.fit(x,y)
    
    # Testing
    x,y = util.load_dataset(eval_path,add_intercept=False)
    h = clf.predict(x)
    h = np.reshape(h,h.size)
    
    num_errs = sum(abs(h-y))
    print('Number of errors in the prediction: ' + str(num_errs))
    
    # Post-processing: separate correct predictions from incorrect predictions
    deltas = h-y
    idx_one_one = np.logical_and(np.asarray(deltas == 0),np.asarray(y == 1)).nonzero()[0] # y is 1, prediction is 1
    idx_one_zero = np.logical_and(np.asarray(deltas == -1),np.asarray(y == 1)).nonzero()[0] # y is 1, prediction is 0
    idx_zero_zero = np.logical_and(np.asarray(deltas == 0),np.asarray(y == 0)).nonzero()[0] # y is 0, prediction is 0
    idx_zero_one = np.logical_and(np.asarray(deltas == 1),np.asarray(y == 0)).nonzero()[0] # y is 0, prediction is 1
    if idx_one_one.size+idx_one_zero.size+idx_zero_zero.size+idx_zero_one.size != y.size:
        raise Exception('Testing error')
    
    # Post-processing: calculate hypotheses
    x1 = np.linspace(min(x[:,0]),max(x[:,0]),num=100) # columns (x1)
    x2 = np.linspace(min(x[:,1]),max(x[:,1]),num=100) # rows (x2)
    g = np.zeros((len(x1),len(x2)))
    # Loop over rows
    for xdx in range(len(x2)):
        x_g = np.concatenate((np.reshape(x1,(x1.size,1)),x2[xdx]*np.ones((x1.size,1))),axis=1)
        x_g = x_g.transpose()
        theta_dot_x = np.matmul(clf.theta.transpose(),x_g) + clf.theta0 # dot product of theta and x, <theta,x>. Dimensions: 1 x m
        theta_dot_x[np.where(theta_dot_x < -20)] = -20 # avoid overflow
        theta_dot_x[np.where(theta_dot_x > 20)] = 20
        g_row = 1/(1 + np.exp(-theta_dot_x)) # g(<theta,x>). Dimensions: 1 x m
        g[xdx,:] = g_row
   
    # Plot testing results
    plt.figure()
    if idx_one_one.size > 0:
        plt.scatter(x[idx_one_one,0],x[idx_one_one,1],marker="^",s=200,label="y=1,h=1")
    if idx_zero_zero.size > 0:
        plt.scatter(x[idx_zero_zero,0],x[idx_zero_zero,1],marker="o",s=50,label="y=0,h=0")
    if idx_one_zero.size > 0:
        plt.scatter(x[idx_one_zero,0],x[idx_one_zero,1],marker="^",s=200,label="y=1,h=0")
    if idx_zero_one.size > 0:
        plt.scatter(x[idx_zero_one,0],x[idx_zero_one,1],marker="o",s=50,label="y=0,h=1")

    contour_obj = plt.contour(x1,x2,g,np.linspace(0,1,11))
    
    plt.title("Testing Results for Logistic Regression",{'fontsize':40})
    plt.xlabel("x_1",{'fontsize':30})
    plt.ylabel("x_2",{'fontsize':30})
    plt.clabel(contour_obj,contour_obj.levels, inline=True, fontsize=15)
    plt.legend(loc="upper left",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    
    return clf


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self,x,y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        
        if y.ndim != 1:
            raise Exception('Incorrect number of dimensions for y')
        if x.ndim != 2:
            raise Exception('Incorrect number of dimensions for x')

        m = x.shape[0] # number of samples
        n = x.shape[1] # number of input features
        
        # Convert x to n*m
        # Rows are features, columns are samples, that is - [column vec 1, column vec 2, ... column vec m]
        x = x.transpose()
        
        if y.size != m:
            raise Exception('The number of samples in x does not match the number of samples in y')
        
        print('GDA learning algorithm')
        print('------------------------------------------\n')
        tstart = time.perf_counter() # in seconds
        
        # MLE
        idx_zeros = np.asarray(y == 0).nonzero()[0]
        idx_ones = np.asarray(y == 1).nonzero()[0]
        phi = len(idx_ones)/m
        mu0 = np.sum(x[:,idx_zeros],axis=1).reshape((n,1))/len(idx_zeros)
        mu1 = np.sum(x[:,idx_ones],axis=1).reshape((n,1))/len(idx_zeros)
        
        mumat = np.zeros((x.shape))
        for ndx in range(n):
            mumat[ndx,idx_zeros] = mu0[ndx][0]
            mumat[ndx,idx_ones] = mu1[ndx][0]
        
        Sigma = np.matmul(x-mumat,(x-mumat).transpose())/m
        Sigma_inv = linalg.inv(Sigma)
        
        # Calculate theta
        self.theta = np.matmul(Sigma_inv,mu1-mu0)
        self.theta0 = 1/2*np.matmul((mu0+mu1).transpose(),np.matmul(Sigma_inv,(mu0-mu1))) - np.log((1-phi)/phi)
        
        tstop = time.perf_counter() # in seconds
        telapse = tstop-tstart
        print('Elapsed time (s) = ' + str(telapse))

    def predict(self,x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        if x.ndim != 2:
            raise Exception('Incorrect number of dimensions for x')

        m = x.shape[0] # number of samples
        n = x.shape[1] # number of input features
        
        # Convert x to n*m
        # Rows are features, columns are samples, that is - [column vec 1, column vec 2, ... column vec m]
        x = x.transpose()
        
        # Calculate hypotheses
        theta_dot_x = np.matmul(self.theta.transpose(),x) + self.theta0 # dot product of theta and x, <theta,x>. Dimensions: 1 x m
        theta_dot_x[np.where(theta_dot_x < -20)] = -20 # avoid overflow
        theta_dot_x[np.where(theta_dot_x > 20)] = 20
        g = 1/(1 + np.exp(-theta_dot_x)) # g(<theta,x>). Dimensions: 1 x m
        h = np.round(g)
        return h

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 2 # data set index, 1 or 2
    #clf = main('../data/ds1_train.csv','../data/ds1_valid.csv','../data/')
    clf = main('../data/ds' + str(dsidx) + '_train.csv','../data/ds' + str(dsidx) + '_valid.csv','../data/')