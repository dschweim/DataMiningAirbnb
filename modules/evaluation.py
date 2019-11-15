from datetime import datetime
from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def calculate_r_squared(y_actual, y_predicted):
    r_squared = r2_score(y_actual, y_predicted)
    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": R_Squared = ", str(r_squared))


def calculate_root_mean_squared_error(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Root Mean Squared Error = ", str(rmse))

