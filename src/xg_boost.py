import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
import os

# Load dataset
diamonds = sns.load_dataset('diamonds')

# Select features and target
features = ['cut', 'color', 'clarity', 'carat', 'depth', 'table', 'x', 'y', 'z']
target = 'price'

# Prepare the features
X = diamonds[features]
y = diamonds[target]

# Apply one-hot encoding to categorical features
X = pd.get_dummies(X, columns=['cut', 'color', 'clarity'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.3,
    "seed": 42
}

# Train model with evaluation set
eval_list = [(dtrain, 'train'), (dtest, 'test')]
evals_result = {}
xg_reg = xgb.train(params, dtrain, num_boost_round=50, evals=eval_list, evals_result=evals_result, verbose_eval=False)

# Extract training and validation RMSE
train_rmse = evals_result['train']['rmse']
test_rmse = evals_result['test']['rmse']

# Predict on test set
y_pred = xg_reg.predict(dtest)

# Calculate evaluation metrics
test_mse = mean_squared_error(y_test, y_pred)
test_rmse_final = mean_squared_error(y_test, y_pred, squared=False)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nBest Model Performance on Test Set:")
print(f'MSE: {test_mse}')
print(f'RMSE: {test_rmse_final}')
print(f'MAE: {test_mae}')
print(f'R2: {test_r2}')

# Plot RMSE over iterations
os.makedirs('results', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_rmse) + 1), train_rmse, label='Train RMSE')
plt.plot(range(1, len(test_rmse) + 1), test_rmse, label='Test RMSE')
plt.xlabel('Number of Iterations')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Iterations')
plt.legend()
plt.savefig('results/xgboost_rmse_plot.png')
plt.close()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('results/xg_boost_residual_plot.png')
plt.close()

# Feature importance plot
plt.figure(figsize=(10, 6))
xgb.plot_importance(xg_reg)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('results/xg_boost_feature_importance.png')
plt.close()