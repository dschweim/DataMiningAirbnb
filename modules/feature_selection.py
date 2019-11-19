import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def feature_selection(df):
    # Manual feature selection
    print("here")
    selected_features = ['accommodates', 'bedrooms', 'beds', 'extra_people', 'guests_included', 'latitude', 'longitude',
                         'room_type', 'price']
    reduced_df = df.loc[:, df.columns.intersection(selected_features)]
    return reduced_df


def univariate_selection(df, number_of_features):
    # Split features and labels
    print("got here")
    features = df.drop('max_price', 1)
    labels = df['max_price']
    print("got here 2")

    # Select k best features using SelectKBest (k = 5)
    best_features = SelectKBest(score_func=chi2, k=number_of_features)
    fit = best_features.fit(features, labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(features.columns)
    print("got here 3")
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print("got here 4")
    print(featureScores.nlargest(10, 'Score'))  # print 10 best features
    return best_features
