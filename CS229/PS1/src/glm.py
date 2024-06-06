# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:10:28 2024

Create generic class for generalized linear models

@author: Ryan Tsai
"""

import numpy as np
import util
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
from numpy import linalg
import scipy.special

from linear_model import LinearModel

class GeneralizedLinearModel(LinearModel):
    """Generalized linear model

    Example usage:
        > clf = GeneralizedLinearModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    
    def __init__(self,dist=None,step_size=0.000001,max_iter=40000,eps=1e-5,theta_0=None,verbose=False):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.dist = dist

    def fit(self,x,y,solver='newton'):
        """GLM training using gradient ascent or Newton's method

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
            dist: Distribution (e.g. Bernoulli, Gaussian, Poisson, etc.)
            solver: 'newton' (default) for Newton's method. 'bgd' for batch gradient descent. 'sgd' for stochastic gradient descent.
        """
        
        if self.dist == None:
            raise Exception('No distribution specified')
        
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
        self.theta = np.zeros((n,1))
        
        # Iteration algorithm
        if solver == 'newton':
            print('Generalized linear model with ' + self.dist + ' distribution and Newton\'s method')
        else:
            print('Generalized linear model with ' + self.dist + ' distribution using ' + solver + ' with learning rate = ' + str(self.step_size))

        print('------------------------------------------\n')
        tstart = time.perf_counter() # in seconds
        
        if solver == 'sgd':
            mdx = 0 # sample index
        
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
            if solver == 'bgd' or solver == 'newton':
                x_sample = x
                y_sample = y
            elif solver == 'sgd':
                x_sample = x[:,mdx]
                x_sample = np.reshape(x_sample,(x_sample.size,1))
                y_sample = y[mdx]
                
            # Calculate the hypothesis, h = g(<theta,x>). g is the canonical response function.
            eta = np.matmul(self.theta.transpose(),x_sample) # dot product of theta and x, <theta,x>. Dimensions: 1 x m. AKA natural parameter.
            g = self.g(eta)
            
            err = y_sample-g # error
            if solver == 'bgd':
                nabla = np.matmul(x_sample,err.transpose()) # gradient of log likelihood
                updates = self.step_size*nabla # updates for next iteration
            elif solver == 'sgd':
                updates = self.step_size*x_sample*err
            elif solver == 'newton':
                nabla = np.matmul(x_sample,err.transpose()) # gradient of log likelihood
                hess = np.matmul(x_sample,x_sample.transpose())*np.matmul(g,1-g.transpose()) # dimensions: n x n
                hess_inv = linalg.inv(hess)
                updates = np.matmul(hess_inv,nabla)

            # Print debug outputs and check for convergence
            updates_l1_norm = sum(abs(updates))[0]
            if (self.verbose and idx in idxs) or updates_l1_norm < self.eps:
                if self.verbose:
                    # Calculate log likelihood
                    if self.dist == 'bernoulli':
                        b = 1
                        a = np.log(1 + np.exp(eta))
                        ell = np.log(b) + eta*y_sample - a
                    elif self.dist == 'gaussian':
                        b = 1/np.sqrt(2*np.pi)*np.exp(-y**2/2)
                        a = 1/2*eta**2
                        ell = np.log(b) + eta*y_sample - a
                    elif self.dist == 'poisson':
                        #b = 1/scipy.special.factorial(y_sample)
                        #b = np.exp(-scipy.special.gammaln(y_sample+1))
                        a = np.exp(eta)
                        ell = -scipy.special.gammaln(y_sample+1) + eta*y_sample - a
                    else:
                        raise Exception('Distribution not supported')
                      
                    #L = b*np.exp(eta*y_sample - a)
                    #ell = np.log(L)
                    #ell = np.log(b) + eta*y_sample - a
                    ell = np.sum(ell,axis=1)[0] # log likelihood (which we're trying to maximize)
                    
                    # Calculate loss (which we're trying to minimize)
                    if solver == 'bgd' or solver == 'newton':
                        loss = -ell/m
                    elif solver == 'sgd':
                        loss = -ell
                    
                    print('Iteration: ' + str(idx))
                    print('------------------------------------------')
                    print('Theta = ' + str(np.round(self.theta,2)))
                    err_abs_sum = np.sum(abs(err),axis=1)[0]
                    print('Sum of absolute error = ' + str(np.round(err_abs_sum,2)))
                    print('Log likelihood = ' + str(round(ell,2)))
                    print('Loss = ' + str(round(loss,2)))
                    print('Update = ' + str(np.round(updates,2)))
                    print('L1 norm of update vector = ' + str(updates_l1_norm))
                    print('\n')
                    
                    thetas[:,odx] = np.reshape(self.theta,self.theta.size)
                    errs[odx] = err_abs_sum
                    losses[odx] = loss
                    
                    if updates_l1_norm < self.eps:
                        # Truncate output variables
                        idxs[odx] = idx
                        idxs = idxs[0:odx+1]
                        thetas = thetas[:,0:odx+1]
                        errs = errs[0:odx+1]
                        losses = losses[0:odx+1]
                    
                    odx = odx+1
                
                # If L1 norm of the updates is less than epsilon, you're finished
                if updates_l1_norm < self.eps:
                    break                 
            
            # Update theta
            self.theta = self.theta + updates
            if solver == 'sgd':
                mdx = (mdx+1) % m
        
        if self.verbose:
            plt.figure()
            plt.plot(idxs,thetas[0,:],linewidth=5,label='theta_0')
            plt.plot(idxs,thetas[1,:],linewidth=5,label='theta_1')
            plt.plot(idxs,thetas[2,:],linewidth=5,label='theta_2')
            plt.title("GLM: Theta vs. Iteration",{'fontsize':40})
            plt.xlabel("Iteration",{'fontsize':30})
            plt.ylabel("Theta",{'fontsize':30})
            plt.legend(loc="upper right",fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid()
            
            plt.figure()
            plt.plot(idxs,errs,linewidth=5)
            plt.title("GLM: Sum of Absolute Error vs. Iteration",{'fontsize':40})
            plt.xlabel("Iteration",{'fontsize':30})
            plt.ylabel("Sum of Absolute Error",{'fontsize':30})
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid()
            
            plt.figure()
            plt.plot(idxs,losses,linewidth=5)
            plt.title("GLM: Loss vs. Iteration",{'fontsize':40})
            plt.xlabel("Iteration",{'fontsize':30})
            plt.ylabel("Loss",{'fontsize':30})
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid()
            
        tstop = time.perf_counter() # in seconds
        telapse = tstop-tstart
        print('Number of iterations = ' + str(idx))
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
        eta = np.matmul(self.theta.transpose(),x) # dot product of theta and x, <theta,x>. Dimensions: 1 x m
        g = self.g(eta)
        
        h = g
        return h
    
    def g(self,eta):
        
        if self.dist == 'bernoulli':
            g = 1/(1 + np.exp(-eta)) # g(<theta,x>). Dimensions: 1 x m
        elif self.dist == 'gaussian':
            g = eta
        elif self.dist == 'poisson':
            g = np.exp(eta)
        else:
            raise Exception('Distribution not supported')
        
        return g