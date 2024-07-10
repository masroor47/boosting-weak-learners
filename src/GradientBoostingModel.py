import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from typing import List, Callable, Optional


class GradientBoostingModel:
    '''
    Gradient Boosting Model
    '''
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 base_learner: Optional[Callable]=None,
                 neg_grad_objective_function: Optional[Callable]=None,
                 n_iterations: int=None,
                 eta: float = 0.3,
                 **kwargs
                 ) -> None:
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
        '''
        self.X = X
        self.y = y
        self.base_learner = base_learner or DecisionTreeRegressor
        self.eta = eta
        self.kwargs = kwargs

        # if base learner is provided, make sure it has the fit and predict methods
        if base_learner:
            assert hasattr(base_learner, 'fit'), 'base_learner must have a fit method'
            assert hasattr(base_learner, 'predict'), 'base_learner must have a predict method'

        self.is_classification = len(np.unique(y)) == 2
        self.n_samples, _ = X.shape

        self.n_iterations = n_iterations or (1000 if self.is_classification else 500)

        if self.is_classification:
            if neg_grad_objective_function is None:
                def get_func():
                    def neg_grad_objective_function(y, y_pred):
                        return y - 1 / (1 + np.exp(-y_pred))
                    return neg_grad_objective_function
                self.neg_grad_objective_function = get_func()

        else:
            if neg_grad_objective_function is None:
                def get_func():
                    def neg_grad_objective_function(y, y_pred):
                        return y - y_pred
                    return neg_grad_objective_function
                self.neg_grad_objective_function = get_func()

    def fit(self) -> List[Callable]:
        '''
        Fit the model

        Returns
        -------
        List[Callable]
            List of boosting functions
        '''
        boosting_functions = []
        y_pred = np.zeros(self.n_samples)

        for _ in tqdm(range(self.n_iterations)):
            gradient = self.neg_grad_objective_function(self.y, y_pred)
            boosting_functions.append(self.base_learner().fit(self.X, gradient))
            y_pred += self.eta * boosting_functions[-1].predict(self.X)

        return boosting_functions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict the target

        Parameters
        ----------
        X : np.ndarray
            The input data matrix

        Returns
        -------
        np.ndarray
            The predicted target
        '''
        return sum(g.predict(X) for g in self.fit())

        