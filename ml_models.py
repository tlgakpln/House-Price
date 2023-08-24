import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import progressbar
import preprocess
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# Load your data
data = preprocess.main()
# Define X (features) and y (target)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with their respective hyperparameter search space
models = [
    ("Linear Regression", LinearRegression(), {}),
    ("Decision Tree Regression", DecisionTreeRegressor(), {'max_depth': [None, 10, 20]}),
    ("Random Forest Regression", RandomForestRegressor(), {'n_estimators': [100, 200]}),
    ("LightGBM", LGBMRegressor(), {'num_leaves': [31, 50], 'max_depth': [-1, 10,]}),
    ("XGBoost", XGBRegressor(), {'n_estimators': [100, 200], 'max_depth': [3, 5]}),
    ("CatBoost", CatBoostRegressor(verbose=False), {'n_estimators': [100, 200]})
]

# Initialize results list
results = []

# Initialize best test R2 score and best model
best_test_r2 = float('-inf')
best_model = None


for model_name, base_model, param_dist in models:
    print(f"Running {model_name}")

    tqdm_iterator = tqdm(range(10))  # Create a tqdm iterator
    # Hyperparameter tuning using RandomizedSearchCV
    for iteration in tqdm_iterator:
        random_search = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=10, cv=5,
                                           scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        tqdm_iterator.set_description(f"Progress: {iteration}")
    tqdm_iterator.close()  # Close the tqdm iterator when done

    best_model = random_search.best_estimator_

    # Evaluate cross-validation performance
    cv_rmse_mean = np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean())
    cv_r2_mean = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1).mean()

    # Train stacking regressor
    stacking_regressor = StackingRegressor(estimators=[(model_name, best_model)], final_estimator=LinearRegression())
    stacking_regressor.fit(X_train, y_train)

    # Evaluate test performance
    y_pred = stacking_regressor.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)

    results.append((model_name, cv_rmse_mean, cv_r2_mean, r2_test, mse_test, mae_test))
    print("----------------------------------------------------------------------------")

    # Update the best model if current model has a better test R2 score
    if r2_test > best_test_r2:
        best_test_r2 = r2_test
        best_model = stacking_regressor
# Save the best model to a file using joblib
joblib.dump(best_model, 'best_model.pkl')

# Create a DataFrame to display results
results_df = pd.DataFrame(results, columns=['Model', 'CV RMSE Mean', 'CV R2 Mean', 'Test R2', 'Test MSE', 'Test MAE'])

# Print results
print("Model Evaluation Results:")
print(results_df)

# Create a bar plot of test R2 scores
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Test R2'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Test R2 Score')
plt.title('Test R2 Score of Different Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
