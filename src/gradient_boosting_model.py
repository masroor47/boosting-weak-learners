import numpy as np

from sklearn.tree import DecisionTreeRegressor


def gradient_boosting_model(X: np.ndarray, 
                            y: np.ndarray, 
                            g_base_learner=None, 
                            neg_grad_objective_function=None,
                            M: int=None,
                            eta: float = 0.3,
                            **kwargs):
    '''
    Gradient Boosting Model

    Parameters
    ----------
    X : np.ndarray
        The input data matrix
    y : np.ndarray
        The target vector
    g_base_learner : function
        The base learner function
    neg_grad_objective_function : function
        The negative gradient of the objective function
    M : int
        The number of iterations
    eta : float
        The learning rate
    kwargs : dict
        Additional arguments for the base learner
    '''
    
    assert isinstance(X, np.ndarray), 'X must be a numpy array'
    assert len(X.shape) == 2, 'X must be a matrix'
    assert isinstance(y, np.ndarray), 'y must be a numpy array'
    assert len(y.shape) == 1, 'y must be a vector'
    assert g_base_learner is None or callable(g_base_learner), 'g_base_learner must be a function'
    assert neg_grad_objective_function is None or callable(neg_grad_objective_function), 'neg_grad_objective_function must be a function'
    assert M is None or isinstance(M, int), 'M must be an integer'
    assert eta is None or isinstance(eta, float), 'eta must be a float'

    n, p = X.shape

    # if base learner is None, it should be a decision tree
    if g_base_learner is None:
        g_base_learner = DecisionTreeRegressor

    # see if it's a regression or classification problem
    if len(np.unique(y)) == 2:
        is_classification = True
        M = 1000 if M is None else M
        if neg_grad_objective_function is None:
            # negative gradient for binary classification
            neg_grad_objective_function = lambda y, g_list: y - 1 / (1 + np.exp(-np.sum([g.predict(X) for g in g_list], axis=0)))
        
        g_0 = lambda X_star, y: np.full(X_star.shape[0], np.exp(np.mean(y)) / (1 + np.exp(np.mean(y))))
    else:
        is_classification = False
        M = 500 if M is None else M
        if neg_grad_objective_function is None:
            # negative gradient for mean squared error
            neg_grad_objective_function = lambda y, y_hat: 2 * (y - y_hat)

        g_0 = lambda X_star, y: np.full(X_star.shape[0], np.mean(y))


    g_list = [g_0]
    def create_gm(m):
        def gm(X_star):
            y_hat_m = np.zeros(n)
            for k in range(m):
                y_hat_m += g_list[k](X)
            
            neg_gradient_m = neg_grad_objective_function(y, y_hat_m)
            g_tilde_m = g_base_learner(X, neg_gradient_m)
            return g_list[m-1](X_star) + eta * g_tilde_m(X_star)
        return gm

    for m in range(M):
        g_list.append(create_gm(m))

    return g_list