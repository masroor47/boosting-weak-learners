import numpy as np

from sklearn.tree import DecisionTreeRegressor
from typing import List, Callable, Optional


def gradient_boosting_model(
    X: np.ndarray, 
    y: np.ndarray, 
    base_learner: Optional[Callable]=None, 
    neg_grad_objective_function: Optional[Callable]=None,
    n_iterations: int=None,
    eta: float = 0.3,
    **kwargs
) -> List[Callable]:
    '''
    Gradient Boosting Model

    Parameters
    ----------
    X : np.ndarray
        The input data matrix
    y : np.ndarray
        The target vector
    base_learner : function
        The base learner function
    neg_grad_objective_function : function
        The negative gradient of the objective function
    n_iterations : int
        The number of iterations
    eta : float
        The learning rate
    kwargs : dict
        Additional arguments for the base learner

    Returns:
    --------
    List[Callable]
        List of boosting functions
    '''
    
    assert isinstance(X, np.ndarray), 'X must be a numpy array'
    assert len(X.shape) == 2, 'X must be a matrix'
    assert isinstance(y, np.ndarray), 'y must be a numpy array'
    assert len(y.shape) == 1, 'y must be a vector'
    assert base_learner is None or callable(base_learner), 'g_base_learner must be a function'
    assert neg_grad_objective_function is None or callable(neg_grad_objective_function), 'neg_grad_objective_function must be a function'
    assert n_iterations is None or isinstance(n_iterations, int), 'M must be an integer'
    assert eta is None or isinstance(eta, float), 'eta must be a float'

    n_iterations, _ = X.shape

    # if base learner is None, it should be a decision tree
    base_learner = base_learner or DecisionTreeRegressor
    is_classification = len(np.unique(y)) == 2



    # see if it's a regression or classification problem
    if is_classification:
        if neg_grad_objective_function is None:
            # negative gradient for binary classification
            def neg_grad_objective_function(y, y_hat):
                return y - 1 / (1 + np.exp(-y_hat))
        g_0 = lambda X_star, y: np.full(X_star.shape[0], np.exp(np.mean(y)) / (1 + np.exp(np.mean(y))))

    else:
        if neg_grad_objective_function is None:
            # negative gradient for mean squared error
            def neg_grad_objective_function(y, y_hat):
                return 2 * (y - y_hat)
        g_0 = lambda X_star, y: np.full(X_star.shape[0], np.mean(y))


    g_list = [g_0]
    
    for m in range(n_iterations):
        y_hat = sum(g(X) for g in g_list)
        neg_gradient = neg_grad_objective_function(y, y_hat)
        
        g_m = base_learner(**kwargs)
        g_m.fit(X, neg_gradient)
        
        g_list.append(lambda X_star, g_m=g_m: eta * g_m.predict(X_star))

    return g_list