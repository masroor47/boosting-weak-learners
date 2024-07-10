import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from typing import List, Callable, Optional

class GradientBoostingModel:
    def __init__(self,
                 base_learner: Optional[Callable]=None,
                 neg_grad_objective_function: Optional[Callable]=None,
                 n_iterations: int=None,
                 eta: float = 0.3,
                 **kwargs
                 ) -> None:
        self.base_learner = base_learner or DecisionTreeRegressor
        self.eta = eta
        self.kwargs = kwargs
        self.n_iterations = n_iterations
        self.neg_grad_objective_function = neg_grad_objective_function
        self.boosting_functions = []
        self.is_fitted = False

    def _init_objective_function(self, y):
        self.is_classification = len(np.unique(y)) == 2
        self.n_iterations = self.n_iterations or (1000 if self.is_classification else 500)

        if self.is_classification:
            if self.neg_grad_objective_function is None:
                self.neg_grad_objective_function = lambda y, y_pred: y - 1 / (1 + np.exp(-y_pred))
            self.initial_prediction = lambda X: np.full(X.shape[0], np.exp(np.mean(y)) / (1 + np.exp(np.mean(y))))
        else:
            if self.neg_grad_objective_function is None:
                self.neg_grad_objective_function = lambda y, y_pred: y - y_pred
            self.initial_prediction = lambda X: np.full(X.shape[0], np.mean(y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._init_objective_function(y)
        self.boosting_functions = [self.initial_prediction]
        
        y_pred = self.initial_prediction(X)
        
        # learning rate decay
        self.eta_list = [self.eta / (i + 1) for i in range(self.n_iterations)]
        # linear decay
        self.eta_list = [self.eta - i * (self.eta / self.n_iterations) for i in range(self.n_iterations)]
        # linear decay with minimum
        self.eta_list = [max(self.eta - i * (self.eta / self.n_iterations), 0.03) for i in range(self.n_iterations)]

        import matplotlib.pyplot as plt
        plt.plot(self.eta_list)
        plt.show()

        
        for _ in tqdm(range(self.n_iterations), desc="Boosting Progress", unit="iteration"):
            gradient = self.neg_grad_objective_function(y, y_pred)
            
            base_model = self.base_learner(**self.kwargs)
            base_model.fit(X, gradient)
            
            curr_eta = self.eta_list.pop(0)
            new_func = lambda X_star, model=base_model: curr_eta * model.predict(X_star)
            self.boosting_functions.append(new_func)
            
            y_pred += new_func(X)
        
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' with appropriate arguments before using 'predict'.")
        return sum(g(X) for g in self.boosting_functions)