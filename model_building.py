# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:16:35 2024

@author: mzing
"""

import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

df = pd.read_csv('data_eda.csv')

#choose relevant columns
df.columns

df_model = df[['avg_salary', 'job_simp','seniority', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector',
               'Revenue', 'age', 'job_state', 'job_city', 'desc_len', 'hourly','remote','bachelors', 'masters',
               'PhD', 'statistics', 'mathematics', 'computer Science', 'economics', 'python',
               'sql', 'java', 'R', 'spark', 'power_bi', 'sas', 'aws', 'excel', 'tableau']]

#get dummy data
df_dum =  pd.get_dummies(df_model)

df_dum = df_dum.astype(int)

#train test split
from sklearn.model_selection import train_test_split

X =df_dum.drop('avg_salary', axis =1)
y =df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Multiple linear Regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

np.mean(cross_val_score(linear_reg, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

np.mean(cross_val_score(gb_model, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

np.mean(cross_val_score(dt_model, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

# Lasso Regression
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.26)
lasso_reg.fit(X_train,y_train)

np.mean(cross_val_score(lasso_reg, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/50)
    lml = Lasso(alpha = (i/50))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err =  pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()
rf_model.fit(X_train,y_train)

np.mean(cross_val_score(rf_model, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))


# XGBoost
from xgboost import XGBRegressor

xg_model = XGBRegressor()
xg_model.fit(X_train, y_train)

np.mean(cross_val_score(xg_model, X_train, y_train, scoring= 'neg_mean_absolute_error', cv=3))

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error

#tune models GridSearchCv
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Define hyperparameter grids
param_grids = {
    'Linear Regression': {'copy_X': [True, False],'fit_intercept': [True, False], 'positive': [True, False]},
    'Lasso Regression': {'alpha': [0.26]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200],
                          'learning_rate': [0.01, 0.1, 0.3],
                          'max_depth': [3, 4, 5]},
    'Random Forest': {'n_estimators': [50, 100, 200],
                      'max_depth': [5, 10, 15],
                      'min_samples_split': [2, 5, 10]},
    'Decision Tree': {'max_depth': [3, 5, 7],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]},
    'XGBoost': {'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 6, 10]},  
}

# 10-Fold cross validatio
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning
for model_name, model in models.items():
    print(f"Hyperparameter tuning for {model_name}:")
    grid_search = GridSearchCV(model, param_grid=param_grids[model_name],
                               scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    mae = mean_absolute_error(grid_search.predict(X_test), y_test)
    print(f"Best parameters: {best_params}")
    print(f"Best MAE on validation set: {best_score}")
    print(f"MAE on test set: {mae}\n")


# Ensemble using stacking(with best models)
from sklearn.ensemble import StackingRegressor

# Initialize base models
lasso_model = Lasso(alpha=0.26)
rf_model = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=200)

# Create the stacking regressor
estimators = [('lasso', lasso_model), ('rf', rf_model)]
stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions on the test set
stacking_pred = stacking_model.predict(X_test)

# Calculate MAE for the stacking ensemble
stacking_mae = mean_absolute_error(y_test, stacking_pred)
print(f"Stacking MAE on test set: {stacking_mae}")


# Averaging

lasso_model = Lasso(alpha=0.26)
rf_model = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=200)

# Train all models on the training dataset
lasso_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

pred_1 = lasso_model.predict(X_test)
pred_2 = rf_model.predict(X_test)

pred_final = (pred_1 + pred_2) / 2
average_mae = mean_absolute_error(y_test, pred_final)
print(f"Average MAE on test set: {average_mae}")











