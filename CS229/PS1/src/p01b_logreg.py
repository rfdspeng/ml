import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y, solver='newton', alpha=0.01):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
            solver: 'newton' (default) for Newton's method. 'gradient descent' for gradient descent
            alpha: learning rate. Only applicable for solver == 'gradient descent'
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        
        if y.ndim != 1:
            raise Exception('Incorrect number of dimensions for y')
        if x.ndim != 2:
            raise Exception('Incorrect number of dimensions for x')

        m = x.shape[0] # number of samples
        n = x.shape[1] # number of input features
        
        # Convert x to n*m (rows are features, columns are samples)
        x = x.transpose()
        
        if y.size != m:
            raise Exception('The number of samples in x does not match the number of samples in y')
        
        # Gradient descent solver
        if solver == 'gradient descent':
            # Initialize theta
            theta = np.zeros((n,1))
            updates = np.zeros((n,1))
            
            # Theta iterations
            for idx in range(10):
            
                # Loop over input features and calculate updates for next iteration
                for ndx in range(n):
                    # g = 1/(1 + e^(-theta.T*x))
                    # theta.T is 1*n
                    # x is n*m
                    # g is 1*m (calculated for each sample)
                    g = 1/(1 + np.exp(-theta.transpose()*x))
                    
                    # Update calculation
                    update = alpha*sum((y - g)*x[ndx,:])
                    updates[ndx] = update
                
                # Update theta
                for ndx in range(n):
                    theta[ndx] = theta[ndx] + updates[ndx]
            
            self.theta = theta
                
        elif solver == 'newton':
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
