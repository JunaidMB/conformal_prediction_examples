from utils import *

# Read Data
drybeans_df = pd.read_csv("./data/drybeans_multiclass.csv")

# Split Data
## Partition into Features and Target
drybeans_features = drybeans_df.drop(['Class'], axis =1)
drybeans_target = drybeans_df.Class.ravel()

## Apply Label Encoding to Target Variable
le = preprocessing.LabelEncoder()
drybeans_target_label_encoded = le.fit_transform(drybeans_target)

drybeans_label_encoding_dict = dict(zip(drybeans_target, drybeans_target_label_encoded))

## Partition into training and test split
drybeans_features_train, drybeans_features_test, drybeans_target_train, drybeans_target_test = train_test_split(drybeans_features, drybeans_target_label_encoded, test_size = 0.25, random_state=123)

# Create Model
rf = RandomForestClassifier(random_state=123)

# Train and Tune model 
## Define Scoring functions
scoring = {"Accuracy": make_scorer(accuracy_score), "MEA": make_scorer(mean_absolute_error), 'roc_auc': make_scorer(roc_auc_score, multi_class = "ovo", needs_proba = True)}

## Random search of parameters, using 5 fold cross validation
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

random_grid = dict(n_estimators = n_estimators, max_depth = max_depth,  min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)

rf_classifier_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 5, scoring = scoring, refit = "roc_auc", verbose=2, random_state=123, n_jobs = -1)

## Fit the Random Forest model
rf_classifier_random.fit(drybeans_features_train, drybeans_target_train)

## Extract best Random Forest model
rf_best_model = rf_classifier_random.best_estimator_
rf_best_model_params = rf_classifier_random.best_params_

## View performance metrics across all hyperparameter permutations
cv_results_df = pd.DataFrame(rf_classifier_random.cv_results_)

# Evaluate Model on Training Data
## Predictions on train set
y_train_predictions = rf_best_model.predict(drybeans_features_train)

## Confusion Matrix on Test Set
confusion_matrix(drybeans_target_train, y_train_predictions)

## Classification Report on Test Set
print(classification_report(drybeans_target_train, y_train_predictions))

## AUC on Train Set
### Multiclass
roc_auc_score(drybeans_target_train, rf_best_model.predict_proba(drybeans_features_train), multi_class='ovo')

# Generate Prediction Sets on Test Set
mapie_aps = MapieClassifier(estimator = rf_best_model, cv="prefit", method="cumulated_score")

mapie_aps.fit(drybeans_features_test, drybeans_target_test)
y_test_predictions, y_test_aps_predictionset = mapie_aps.predict(drybeans_features_test, alpha=[0.05], include_last_label=True)

## Express as a dataframe
pred_set_list = y_test_aps_predictionset.tolist()
y_ps_classes = [list( np.where(i)[0] ) for i in pred_set_list]

test_predictions_df = pd.DataFrame({'y_test': drybeans_target_test, 'y_prediction_test': y_test_predictions, 'y_prediction_set': y_ps_classes})

# Evaluate Model on Test Data
## Confusion Matrix on Test Set
confusion_matrix(drybeans_target_test, y_test_predictions)

## Classification Report on Test Set
print(classification_report(drybeans_target_test, y_test_predictions))

## AUC on Test Set
### Multiclass
roc_auc_score(drybeans_target_test, rf_best_model.predict_proba(drybeans_features_test), multi_class='ovo')

