import itertools
from math import sqrt

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def baseline_prediction(x_train, x_test, y_train, y_test):
    dummy_mean = DummyRegressor(strategy='mean')
    dummy_mean.fit(x_train, y_train)
    predictions = dummy_mean.predict(x_test)
    r_squared = dummy_mean.score(x_test, y_test)
    print("Root Mean Squared Error :", str(round(sqrt(mean_squared_error(y_test, predictions)), 2)))
    print("R2 :", str(round(r_squared, 4)))


def calculate_mean_absolute_error(y_actual, y_predicted):
    mean_abs_error = mean_absolute_error(y_actual, y_predicted)
    return mean_abs_error


def calculate_root_mean_squared_error(y_actual, y_predicted):
    root_mean_squared_error = sqrt(mean_squared_error(y_actual, y_predicted))
    return root_mean_squared_error


def calculate_r_squared(y_actual, y_predicted):
    r_squared = r2_score(y_actual, y_predicted)
    return r_squared


def generate_feature_combinations(x):
    feature_combinations = []
    for index, column in enumerate(x.columns, start=1):
        if index > 0:
            combinations = itertools.combinations(x.columns, index)
            for combination in combinations:
                tmp = []
                for i in range(0, index):
                    tmp.append(str(combination[i]))
                feature_combinations.append(tmp)
    return feature_combinations
