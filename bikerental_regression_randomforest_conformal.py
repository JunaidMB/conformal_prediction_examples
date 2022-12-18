from random import random
from utils import *

# Read Data
bike_data = pd.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/ml-basics/master/data/daily-bike-share.csv')
bike_data.head()

# Split Data
## Partition into Features and Target
bike_features, bike_labels = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']], bike_data['rentals'].values
print('Features:', bike_features[:10], '\nLabels:', bike_labels[:10], sep='\n')

## Partition into training and test split
bike_features_train, bike_features_test, bike_target_train, bike_target_test = train_test_split(bike_features, bike_labels, test_size = 0.25, random_state=123)

# Create Model
rf = RandomForestRegressor(random_state=123)

# Train and Tune model
## Define Scoring functions
scoring = {"MAE": make_scorer(mean_absolute_error), "MSE": make_scorer(mean_squared_error), "MAPE": make_scorer(mean_absolute_percentage_error)}

## Random search of parameters, using 5 fold cross validation
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

random_grid = dict(n_estimators = n_estimators, max_depth = max_depth,  min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)

# Random search of parameters, using 5 fold cross validation, 
rf_regressor_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 5, scoring = scoring, refit = "MAE", verbose=2, random_state=123, n_jobs = -1)

## Fit the Random Forest model
rf_regressor_random.fit(bike_features_train, bike_target_train)

# Extract best Random Forest model
rf_best_model = rf_regressor_random.best_estimator_
rf_best_model_params = rf_regressor_random.best_params_

## View performance metrics across all hyperparameter permutations
cv_results_df = pd.DataFrame(rf_regressor_random.cv_results_)

# Evaluate Model on Training Data
## Predictions on train set
y_train_predictions = rf_best_model.predict(bike_features_train)

## Mean Square Error
mean_squared_error(bike_target_train, y_train_predictions)

## Mean Absolute Error
mean_absolute_error(bike_target_train, y_train_predictions)

## Mean Absolute Percentage Error (MAPE)
mean_absolute_percentage_error(bike_target_train, y_train_predictions)

# Evaluate on Test Data
## Predictions on test set
y_test_predictions = rf_best_model.predict(bike_features_test)

## Mean Square Error
mean_squared_error(bike_target_test, y_test_predictions)

## Mean Absolute Error
mean_absolute_error(bike_target_test, y_test_predictions)

## Mean Absolute Percentage Error (MAPE)
mean_absolute_percentage_error(bike_target_test, y_test_predictions)

# Generate Conformal Prediction Intervals
alpha = [0.05]
mapie = MapieRegressor(rf_best_model, method="plus")
mapie.fit(bike_features_test, bike_target_test)
y_pred, y_pis = mapie.predict(bike_features_test, alpha=alpha)

y_pred[0]
y_pis[0]
## Express as a dataframe
pred_set_list = y_pis.tolist()
pred_intervals = []
for idx,_ in enumerate(pred_set_list):
    pred_intervals.append( [i[0] for i in pred_set_list[idx]] )


test_predictions_df = pd.DataFrame({'y_test': bike_target_test, 'y_prediction_test': y_pred, 'y_prediction_set': pred_intervals})



