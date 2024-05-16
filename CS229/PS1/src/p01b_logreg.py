import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    
    # add_intercept=True means adding x_0=1 as the first column
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    #x_train, y_train = util.load_dataset(train_path)
    
    # Plot data
    idx_ones = np.where(y_train == 1)[0]
    idx_zeros = np.where(y_train == 0)[0]
    plt.figure()
    plt.scatter(x_train[idx_ones,1],x_train[idx_ones,2],marker="^",s=200,label="y=1")
    plt.scatter(x_train[idx_zeros,1],x_train[idx_zeros,2],marker="o",s=50,label="y=0")
    plt.title("Binary Data for Logistic Regression",{'fontsize':40})
    plt.xlabel("x_1",{'fontsize':30})
    plt.ylabel("x_2",{'fontsize':30})
    plt.legend(loc="upper right",fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()    
    
    clf = LogisticRegression()
    clf.max_iter = 400000
    
    # Learning rate (alpha) for gradient descent
    clf.step_size = 0.000001 # this step size leads to convergence for batch gradient descent    
    clf.fit(x_train, y_train, solver='bgd')
    
    
    #clf.predict(x_eval)
    return clf


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y, solver='newton'):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
            solver: 'newton' (default) for Newton's method. 'bgd' for batch gradient descent. 'sgd' for stochastic gradient descent.
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
        
        # Initialize theta
        if self.theta == None:
            self.theta = np.zeros((n,1))
        
        # Iteration algorithm
        if solver == 'bgd': # batch gradient descent
            print('Logistic regression using batch gradient descent with learning rate = ' + str(self.step_size))
            print('------------------------------------------\n')
            
            # Data for plotting
            if self.verbose:
                niter = 10
                idxs = round(self.max_iter/niter)*np.array(range(niter))
                idxs = np.append(idxs,self.max_iter-1)
                thetas = np.zeros((self.theta.size,idxs.size)) # thetas. Dimensions: n x idxs
                errs = np.zeros(idxs.shape) # sum(abs(y-g))
                losses = np.zeros(idxs.shape) # log likelihood
                odx = 0

            # Theta iterations
            updates = np.zeros((n,1))
            for idx in range(self.max_iter):
                # Calculate the hypothesis, h = g(<theta,x>). g is the sigmoid function.
                theta_dot_x = np.matmul(self.theta.transpose(),x) # dot product of theta and x, <theta,x>. Dimensions: 1 x m
                theta_dot_x[np.where(theta_dot_x < -20)] = -20 # avoid overflow
                theta_dot_x[np.where(theta_dot_x > 20)] = 20
                g = 1/(1 + np.exp(-theta_dot_x)) # g(<theta,x>). Dimensions: 1 x m
                err = y-g # error
                
                if self.verbose and idx in idxs:
                    # Calculate loss
                    ell = y*np.log(g) + (1-y)*np.log(1-g)
                    ell = np.sum(ell,axis=1)[0] # likelihood (which we're trying to maximize)
                    loss = -ell/m # loss (which we're trying to minimize)
                    
                    print('Iteration: ' + str(idx))
                    print('------------------------------------------')
                    print('Theta = ' + str(np.round(self.theta,2)))
                    err_abs_sum = np.sum(abs(err),axis=1)[0]
                    print('Sum of absolute error = ' + str(np.round(err_abs_sum,2)))
                    print('Log likelihood = ' + str(round(ell,2)))
                    print('Loss = ' + str(round(loss,2)))
                    
                    thetas[:,odx] = np.reshape(self.theta,self.theta.size)
                    errs[odx] = err_abs_sum
                    losses[odx] = loss
                    odx = odx+1
                    
                # Calculate updates for next iteration
                updates = self.step_size*np.matmul(x,err.transpose())
                
                # Update theta
                self.theta = self.theta + updates
                
                # If L1 norm of the updates is less than epsilon, you're finished
                updates_l1_norm = sum(abs(updates))[0]
                if updates_l1_norm < self.eps:
                    if self.verbose:
                        # Truncate output variables
                        bmask = idxs <= idx
                        idxs = idxs[bmask]
                        thetas = thetas[:,bmask]
                        errs = errs[bmask]
                        losses = losses[bmask]
                        
                        # Add last iteration
                        
                        # Calculate loss
                        ell = y*np.log(g) + (1-y)*np.log(1-g)
                        ell = np.sum(ell,axis=1)[0] # likelihood (which we're trying to maximize)
                        loss = -ell/m # loss (which we're trying to minimize)
                        
                        err_abs_sum = np.sum(abs(err),axis=1)[0]
                        
                        idxs = np.append(idxs,idx)
                        thetas = np.append(thetas,self.theta,axis=1)
                        errs = np.append(errs,err_abs_sum)
                        losses = np.append(losses,loss)
                        
                    break
                
                # Debug messages
                if self.verbose and idx in idxs:
                    print('Update = ' + str(np.round(updates,2)))
                    print('L1 norm of update vector = ' + str(updates_l1_norm))
                    print('\n')
            
            if self.verbose:
                plt.figure()
                plt.plot(idxs,thetas[0,:],linewidth=5,label='theta_0')
                plt.plot(idxs,thetas[1,:],linewidth=5,label='theta_1')
                plt.plot(idxs,thetas[2,:],linewidth=5,label='theta_2')
                plt.title("Logistic Regression: Theta vs. Iteration",{'fontsize':40})
                plt.xlabel("Iteration",{'fontsize':30})
                plt.ylabel("Theta",{'fontsize':30})
                plt.legend(loc="upper right",fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.grid()
                
                plt.figure()
                plt.plot(idxs,errs,linewidth=5)
                plt.title("Logistic Regression: Sum of Absolute Error vs. Iteration",{'fontsize':40})
                plt.xlabel("Iteration",{'fontsize':30})
                plt.ylabel("Sum of Absolute Error",{'fontsize':30})
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.grid()
                
                plt.figure()
                plt.plot(idxs,losses,linewidth=5)
                plt.title("Logistic Regression: Loss vs. Iteration",{'fontsize':40})
                plt.xlabel("Iteration",{'fontsize':30})
                plt.ylabel("Loss",{'fontsize':30})
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.grid()
        
        elif solver == 'sgd': # stochastic gradient descent
            'tbd'
                
        elif solver == 'newton': # Newton's method
            'tbd'
                    
            

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    plt.close('all')
    get_ipython().magic('reset -sf')
    
    main('../data/ds1_train.csv', '../data/ds1_valid.csv', '../data/')