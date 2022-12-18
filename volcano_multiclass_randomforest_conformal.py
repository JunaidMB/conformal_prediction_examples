from utils import *

# Read Data
volcano_df = pd.read_csv("./data/volcano_multiclass.csv")

# Split Data
## Partition into Features and Target
volcano_features = volcano_df.drop(['volcano_type', 'volcano_number'], axis =1)
volcano_target = volcano_df.volcano_type.ravel()

## Apply Label Encoding to Target Variable
le = preprocessing.LabelEncoder()
volcano_target_label_encoded = le.fit_transform(volcano_target)

volcano_label_encoding_dict = dict(zip(volcano_target, volcano_target_label_encoded))

## Partition into training and test split
volcano_features_train, volcano_features_test, volcano_target_train, volcano_target_test = train_test_split(volcano_features, volcano_target_label_encoded, test_size = 0.25, random_state=123)

# Perform Feature Engineering
'''
Numerical Features: 
1. Standardize

Catergorical Features: 
1. Remove infrequent categories 
2. One Hot Encode
'''
num_feat_names, cat_feat_names = cont_cat_split(volcano_features_train, max_card=20, dep_var=["volcano_type", "volcano_number"])

cat_pipeline = Pipeline([
    ('OHE', preprocessing.OneHotEncoder(min_frequency=0.05, sparse_output=False))
])

num_pipeline = Pipeline([
    ('std_scale', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_feat_names),
    ("cat", cat_pipeline, cat_feat_names)
])

volcano_processed_features_train_fit = full_pipeline.fit(volcano_features_train)
volcano_processed_features_train = full_pipeline.transform(volcano_features_train)
volcano_processed_features_test = full_pipeline.transform(volcano_features_test)

## Make a DataFrame version of Train and Test Data
volcano_processed_features_train_df = pd.DataFrame(volcano_processed_features_train)
volcano_processed_features_train_df.columns = full_pipeline.get_feature_names_out()

volcano_processed_features_test_df = pd.DataFrame(volcano_processed_features_test)
volcano_processed_features_test_df.columns = full_pipeline.get_feature_names_out()

## Balance Classes - Using SMOTE oversampling
oversample = SMOTE()
volcano_processed_features_oversampled_train, volcano_target_oversampled_train = oversample.fit_resample(volcano_processed_features_train, volcano_target_train)

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
rf_classifier_random.fit(volcano_processed_features_oversampled_train, volcano_target_oversampled_train)

## Extract best Random Forest model
rf_best_model = rf_classifier_random.best_estimator_
rf_best_model_params = rf_classifier_random.best_params_

## View performance metrics across all hyperparameter permutations
cv_results_df = pd.DataFrame(rf_classifier_random.cv_results_)

# Evaluate Model on Training Data
## Predictions on train set
y_train_predictions = rf_best_model.predict(volcano_processed_features_train)

## Confusion Matrix on Test Set
confusion_matrix(volcano_target_train, y_train_predictions)

## Classification Report on Test Set
print(classification_report(volcano_target_train, y_train_predictions))

## AUC on Train Set
### Multiclass
roc_auc_score(volcano_target_train, rf_best_model.predict_proba(volcano_processed_features_train), multi_class='ovo')

# Evaluate Model on Test Data
## Predictions on test set
y_test_predictions = rf_best_model.predict(volcano_processed_features_test)

## Confusion Matrix on Test Set
confusion_matrix(volcano_target_test, y_test_predictions)

## Classification Report on Test Set
print(classification_report(volcano_target_test, y_test_predictions))

## AUC on Test Set
### Multiclass
roc_auc_score(volcano_target_test, rf_best_model.predict_proba(volcano_processed_features_test), multi_class='ovo')

# Generate Prediction Sets
mapie_aps = MapieClassifier(estimator=rf_best_model, cv="prefit", method="cumulated_score")

mapie_aps.fit(volcano_processed_features_test, volcano_target_test)
alpha = [0.05]
y_pred_aps, y_ps_aps = mapie_aps.predict(volcano_processed_features_test, alpha=alpha, include_last_label=True)

## Express as a dataframe
pred_set_list = y_ps_aps.tolist()
y_ps_classes = [list( np.where(i)[0] ) for i in pred_set_list]

test_predictions_df = pd.DataFrame({'y_test': volcano_target_test, 'y_prediction_test': y_pred_aps, 'y_prediction_set': y_ps_classes})

# Feature Importance

## Permutation Importance
perm_importance = permutation_importance(rf_best_model, volcano_processed_features_test, volcano_target_test, random_state = 123)

sorted_idx = perm_importance.importances_mean.argsort()

permutation_importance_df = pd.DataFrame({'Features': volcano_processed_features_test_df.columns[sorted_idx], 'Permutation_Importance': perm_importance.importances_mean[sorted_idx]})
permutation_importance_df = permutation_importance_df.sort_values(by = ['Permutation_Importance'], ascending = False).reset_index(drop=True)

## SHAP Matrix
shap_explainer = fasttreeshap.TreeExplainer(rf_best_model, algorithm = "auto", n_jobs = -1)
shap_values = shap_explainer(volcano_processed_features_test)

shap_values_lst = []
for key, value in enumerate(list(range(0, shap_values.values.shape[2]))):
    shap_values_lst.append( shap_values.values[:, :, key] )

shap_matrix_dict = dict( zip([f'class_{i}'for i in list(range(0, shap_values.values.shape[2]))], shap_values_lst))