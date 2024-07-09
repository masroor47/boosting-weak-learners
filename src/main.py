import seaborn as sns
from sklearn.model_selection import train_test_split

from gradient_boosting_model import gradient_boosting_model


SEED = 42

if __name__ == '__main__':
    diamonds = sns.load_dataset('diamonds')

    X = diamonds.drop('price', axis=1)
    y = diamonds['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    gbm = gradient_boosting_model(X_train, y_train, eta=0.3)