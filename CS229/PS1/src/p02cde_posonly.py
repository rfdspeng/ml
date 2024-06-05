import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg
import copy

from p01b_logreg import LogisticRegression
from glm import GeneralizedLinearModel

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path,valid_path,test_path,pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD,'c')
    pred_path_d = pred_path.replace(WILDCARD,'d')
    pred_path_e = pred_path.replace(WILDCARD,'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
    
    #logreg_clf = LogisticRegression()
    #logreg_clf.verbose = False; logreg_clf.max_iter = 400000
    logreg_clf = GeneralizedLinearModel(dist='bernoulli')
    
    """ Train and test on the true labels """
    x,y = util.load_dataset(train_path,label_col='t',add_intercept=True)
    logreg_clf.fit(x,y,solver='newton')
    
    x,y = util.load_dataset(test_path,label_col='t',add_intercept=True)
    h = logreg_clf.predict(x)
    h = np.reshape(h,h.size)
    h = np.round(h)
    num_errs = sum(abs(h-y))
    print('Number of errors when training on the true labels: ' + str(num_errs) + ' out of ' + str(len(y)) + ' data points')
    util.plot_custom(x,y,logreg_clf.theta,title="Logreg Trained on True Labels")
    
    """ Train on the database, estimate alpha on the validation data, test on the true labels """
    x,y = util.load_dataset(train_path,label_col='y',add_intercept=True)
    logreg_clf.fit(x,y,solver='newton')
    
    x,y = util.load_dataset(valid_path,label_col='y',add_intercept=True)
    h = logreg_clf.predict(x)
    h = np.reshape(h,h.size)
    alpha = np.mean(h[y == 1])
    plt.figure()
    plt.hist(h[y == 1])
    plt.title("Histogram of Validation Set Predictions")
    
    x,y = util.load_dataset(test_path,label_col='t',add_intercept=True)
    h = logreg_clf.predict(x)
    h = np.reshape(h,h.size)
    h = np.round(h)
    num_errs = sum(abs(h-y))
    print('Number of errors when training on the database: ' + str(num_errs) + ' out of ' + str(len(y)) + ' data points')
    util.plot_custom(x,y,logreg_clf.theta,title="Logreg Trained on Database")
    #util.plot_custom(x,y,logreg_clf.theta,title="Logreg Trained on Database with Correction",alpha=alpha)
    
    logreg_clf_corr = copy.deepcopy(logreg_clf)
    logreg_clf_corr.theta[0,0] = logreg_clf_corr.theta[0,0] + np.log(2/alpha-1)
    h = logreg_clf_corr.predict(x)
    h = np.reshape(h,h.size)
    h = np.round(h)
    num_errs = sum(abs(h-y))
    print('Number of errors when training on the database, with correction: ' + str(num_errs) + ' out of ' + str(len(y)) + ' data points')
    util.plot_custom(x,y,logreg_clf_corr.theta,title="Logreg Trained on Database with Correction")
    
    return logreg_clf,logreg_clf_corr

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    dsidx = 3
    
    clf,clf_corr = main('../data/ds' + str(dsidx) + '_train.csv','../data/ds' + str(dsidx) + '_valid.csv','../data/ds' + str(dsidx) + '_test.csv','../data/')