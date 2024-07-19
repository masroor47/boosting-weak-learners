import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from gradient_boosting_model import gradient_boosting_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from base_learners.linear_base_learner import WeakLinearModel
import os
from datetime import datetime

SEED = 42

def write_results_to_file(model_label, results_text):
    results_dir = os.path.join(os.getcwd(), 'results/text')
    os.makedirs(results_dir, exist_ok=True)  # This will create the directory if it doesn't exist
    file_path = os.path.join(results_dir, f"{model_label.replace(' ', '_').lower()}_results ({datetime.now().strftime("%B %d %Y %H:%M:%S")  }).txt")
    with open(file_path, 'w') as f:
        f.write(results_text)
    
    print(f"Results written to: {file_path}")


if __name__ == '__main__':
    diamonds = sns.load_dataset('diamonds')
    diamonds = pd.get_dummies(diamonds, drop_first=True)

    base_learners = {
        "Linear model": WeakLinearModel,
        "Shallow trees": None, # this set to None as the default fallback in gradient_boosting_model is DecisionTreeRegressor
    }

    X = diamonds.drop('price', axis=1)
    y = diamonds['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    N_ITERATIONS = 50
    for model_label, base_learner in base_learners.items():
        gbm = gradient_boosting_model(X_train, y_train, base_learner=base_learner, n_iterations=N_ITERATIONS, eta=0.3)
        predictions = sum(g(X_test) for g in gbm)
        
        results_text = f'Results for base learner: {model_label}\n'
        mse = mean_squared_error(y_test, predictions)
        results_text += f'MSE: {mse:.4f}\n'
        results_text += f'RMSE: {np.sqrt(mse):.4f}\n'
        mae = mean_absolute_error(y_test, predictions)
        results_text += f'MAE: {mae:.4f}\n'
        r2 = r2_score(y_test, predictions)
        results_text += f'R2: {r2:.4}\n'
        results_text += '---\n'
        
        print(results_text)
        write_results_to_file(model_label, results_text)
