import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg
import copy

from glm import GeneralizedLinearModel
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    
    """ Training """
    x,y = util.load_dataset(train_path,add_intercept=False)
    
    clf = GeneralizedLinearModel(dist='poisson')
    clf.step_size = clf.step_size**2
    clf.max_iter = clf.max_iter*5
    clf.verbose = True
    clf.fit(x,y,solver='bgd')

    """ Testing """
    x,y = util.load_dataset(eval_path,add_intercept=False)
    h = clf.predict(x)
    h = np.reshape(h,h.size)
    #h = np.round(h)
    
    y_rms = np.sqrt(np.mean(y**2))
    h_rms = np.sqrt(np.mean(h**2))
    e_rms = np.sqrt(np.mean((y-h)**2))
    evm = e_rms/y_rms*100
    snr = -20*np.log10(evm/100)
    
    print('EVM (%) = ' + str(evm))
    print('SNR (dB) = ' + str(snr))
    
    plt.figure()
    plt.scatter(y,y,label='y vs. y')
    plt.scatter(y,h,label = 'h vs. y')
    plt.legend(loc="upper left",fontsize=20)
    plt.grid()

class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 4
    clf = main('../data/ds' + str(dsidx) + '_train.csv','../data/ds' + str(dsidx) + '_valid.csv','../data/')