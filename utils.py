from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import MapieClassifier
from mapie.metrics import (classification_coverage_score, classification_mean_width_score)
from fastai.tabular.all import cont_cat_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, mean_absolute_error, make_scorer, mean_squared_error, mean_absolute_percentage_error
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GroupShuffleSplit, GridSearchCV
import shap
from sklearn.calibration import CalibratedClassifierCV
import os
import pickle
import fasttreeshap
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.subsample import Subsample
