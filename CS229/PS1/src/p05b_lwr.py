import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time

from linear_model import LinearModel


def main(train_path,valid_path,test_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x,y = util.load_dataset(train_path,add_intercept=True)
    clf = LocallyWeightedLinearRegression(0.5)
    clf.fit(x,y)
    
    # Load validation set
    x,y = util.load_dataset(valid_path,add_intercept=True)
    
    taus = [0.03,0.05,0.1,0.5,1,10]
    for tdx in range(len(taus)):
        tau = taus[tdx]
        clf.tau = tau
        h = clf.predict(x)
        
        mse = np.mean(np.abs((y-h))**2)
        print('Tau=' + str(tau) + ', MSE=' + str(mse))
        
        plt.figure()
        plt.scatter(clf.x[:,-1],clf.y,marker="o",s=200,label="y")
        plt.scatter(x[:,-1],h,marker="o",s=50,label="h")
        
        plt.title('LWLR, Tau=' + str(clf.tau),{'fontsize':40})
        plt.xlabel("x",{'fontsize':30})
        plt.ylabel("y",{'fontsize':30})
        plt.legend(loc="upper left",fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
    
    x,y = util.load_dataset(test_path,add_intercept=True)
    tau = 0.05
    clf.tau = tau
    h = clf.predict(x)
    mse = np.mean(np.abs((y-h))**2)
    print('Tau=' + str(tau) + ', MSE=' + str(mse))


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self,tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self,x,y):
        """Fit LWR by saving the training set.
        
        """
        
        if y.ndim != 1:
            raise Exception('Incorrect number of dimensions for y')
        if x.ndim != 2:
            raise Exception('Incorrect number of dimensions for x')
        
        # x is m by n
        # Rows are samples, columns are features
        m = x.shape[0] # number of samples
        n = x.shape[1] # number of input features
        
        if y.size != m:
            raise Exception('The number of samples in x does not match the number of samples in y')
        
        self.x = x; self.y = y

    def predict(self,x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        
        x_test = x # testing
        x = self.x # training
        y = self.y # training
        
        # x is m by n
        # Rows are samples, columns are features
        m = x_test.shape[0] # number of samples
        n = x_test.shape[1] # number of input features
        
        hs = np.zeros(m)
        thetas = np.zeros((n,m))
        for sdx in range(m):
            x_curr = x_test[sdx]
            w = np.exp(-np.linalg.norm(x-x_curr,axis=1)**2/2/self.tau**2)
            W = np.diag(w)
            theta = np.matmul(np.linalg.inv(np.matmul(x.transpose(),np.matmul(W,x))),np.matmul(x.transpose(),np.matmul(W,y)))
            h = np.matmul(theta.transpose(),x_curr)
            thetas[:,sdx] = theta
            hs[sdx] = h
        
        return hs

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 5
    clf = main('../data/ds' + str(dsidx) + '_train.csv','../data/ds' + str(dsidx) + '_valid.csv','../data/ds' + str(dsidx) + '_test.csv')
