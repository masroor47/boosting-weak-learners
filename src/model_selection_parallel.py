import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from gradient_boosting_model import gradient_boosting_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

SEED = 42

def evaluate_model(model, X, y):
    predictions = sum(g(X) for g in model)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, rmse, mae, r2

def train_and_evaluate(X_train, y_train, X_val, y_val, n_iterations, eta):
    gbm = gradient_boosting_model(X_train, y_train, n_iterations=n_iterations, eta=eta)
    val_mse, val_rmse, val_mae, val_r2 = evaluate_model(gbm, X_val, y_val)
    return n_iterations, gbm, val_mse, val_rmse, val_mae, val_r2

if __name__ == '__main__':
    # Load and preprocess data
    diamonds = sns.load_dataset('diamonds')
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    X = X.select_dtypes(include=[np.number])

    # Split data into train+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Further split train+validation into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=SEED)  # 0.25 x 0.8 = 0.2

    # Convert to numpy arrays
    X_train, X_val, X_test = map(lambda x: x.to_numpy(), [X_train, X_val, X_test])
    y_train, y_val, y_test = map(lambda x: x.to_numpy(), [y_train, y_val, y_test])

    # Model selection
    # n_iterations_list = [50, 
    #                      75, 
    #                      100, 
    #                      125, 
    #                      140, 
    #                      150, 
    #                      175, 
    #                      200, 
    #                      300, 
    #                      500]
    
    # n_iterations from 25 to 500 in steps of 25
    n_iterations_list = [ 25 * i for i in range(1, 21)]
    eta = 0.3

    best_model = None
    best_val_mse = float('inf')
    val_results = []

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        future_to_iterations = {
            executor.submit(train_and_evaluate, X_train, y_train, X_val, y_val, n_iterations, eta): n_iterations
            for n_iterations in n_iterations_list
        }

        for future in as_completed(future_to_iterations):
            n_iterations, gbm, val_mse, val_rmse, val_mae, val_r2 = future.result()
            val_results.append((n_iterations, val_mse, val_rmse, val_mae, val_r2))
            
            print(f'N iterations: {n_iterations}, Validation MSE: {val_mse}')
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model = gbm

    # Sort results by number of iterations
    val_results.sort(key=lambda x: x[0])

    # Evaluate best model on test set
    test_mse, test_rmse, test_mae, test_r2 = evaluate_model(best_model, X_test, y_test)

    print("\nBest Model Performance on Test Set:")
    print(f'MSE: {test_mse}')
    print(f'RMSE: {test_rmse}')
    print(f'MAE: {test_mae}')
    print(f'R2: {test_r2}')

    # Plot validation results
    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in val_results], [r[1] for r in val_results], marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Number of Iterations')
    plt.savefig('validation_mse_plot.png')
    plt.close()

    # Plot residuals
    test_predictions = sum(g(X_test) for g in best_model)
    residuals = y_test - test_predictions

    plt.figure(figsize=(10, 6))
    plt.scatter(test_predictions, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residual_plot.png')
    plt.close()