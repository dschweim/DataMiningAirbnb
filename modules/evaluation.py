import itertools
from math import sqrt

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def baseline_prediction(features_train, features_test, label_train, label_test):
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(features_train, label_train)
    predictions = dummy_mean.predict(features_test)
    print("Performance of Dummy Regressor (Mean) :", str(sqrt(mean_squared_error(label_test, predictions))))


def calculate_mean_absolute_error(y_actual, y_predicted):
    mean_abs_error = mean_absolute_error(y_actual, y_predicted)
    return mean_abs_error


def calculate_root_mean_squared_error(y_actual, y_predicted):
    root_mean_squared_error = sqrt(mean_squared_error(y_actual, y_predicted))
    return root_mean_squared_error


def calculate_r_squared(y_actual, y_predicted):
    r_squared = r2_score(y_actual, y_predicted)
    return r_squared


def generate_feature_combinations(df):
    feature_combinations = []
    for index, column in enumerate(df.columns, start=1):
        if index > 0:
            combinations = itertools.combinations(df.columns, index)
            for combination in combinations:
                tmp = []
                for i in range(0, index):
                    tmp.append(str(combination[i]))
                feature_combinations.append(tmp)
    return feature_combinations
