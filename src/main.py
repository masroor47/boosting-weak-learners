import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from gradient_boosting_model import gradient_boosting_model


SEED = 42

if __name__ == '__main__':
    diamonds = sns.load_dataset('diamonds')

    X = diamonds.drop('price', axis=1)
    y = diamonds['price']

    # get only numerical columns
    X = X.select_dtypes(include=[np.number])
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    gbm = gradient_boosting_model(X_train, y_train, eta=0.3)
    predictions = gbm[-1](X_test)
    # precition metrics
    # compute SSE, RMSE, MAE, R2
    # plot the residuals
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    print(f'MSE: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, predictions)
    print(f'MAE: {mae}')
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, predictions)
    print(f'R2: {r2}')
