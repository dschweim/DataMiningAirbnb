from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest, SelectFwe
from collections import Counter
# import geopandas as gpd #requires install via conda/pip
# import geoplot as glt #requires install via conda/pip
# from shapely.geometry import Point, polygon