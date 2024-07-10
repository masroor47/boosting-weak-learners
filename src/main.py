import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from gradient_boosting_model import gradient_boosting_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


SEED = 42

if __name__ == '__main__':
    diamonds = sns.load_dataset('diamonds')
    diamonds = pd.get_dummies(diamonds, columns=['cut', 'color', 'clarity'], drop_first=True)

    X = diamonds.drop('price', axis=1)
    y = diamonds['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    N_ITERATIONS = 50
    gbm = gradient_boosting_model(X_train, y_train, n_iterations=N_ITERATIONS, eta=0.3)
    predictions = sum(g(X_test) for g in gbm)

    mse = mean_squared_error(y_test, predictions)
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {np.sqrt(mse):.2f}')
    mae = mean_absolute_error(y_test, predictions)
    print(f'MAE: {mae:.2f}')
    r2 = r2_score(y_test, predictions)
    print(f'R2: {r2:.2f}')
