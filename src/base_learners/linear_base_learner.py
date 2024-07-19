import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tqdm import tqdm

class WeakLinearModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, n_features_to_select=10, noise_level=0.5):
        self.alpha = alpha
        self.n_features_to_select = n_features_to_select
        self.noise_level = noise_level
        self.model = Ridge(alpha=self.alpha)
        self.selected_features = None

    def fit(self, X, y):
        # Randomly select a subset of features
        n_features = X.shape[1]
        self.selected_features = np.random.choice(n_features, self.n_features_to_select, replace=False)
        X_subset = X[:, self.selected_features]
        self.model.fit(X_subset, y)
        return self

    def predict(self, X):
        X_subset = X[:, self.selected_features]
        predictions = self.model.predict(X_subset)
        noise = np.random.normal(0, self.noise_level, size=predictions.shape)
        return predictions + noise
