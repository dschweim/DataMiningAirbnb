import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from modules.feature_selection import feature_selection
from modules.feature_selection import univariate_selection


def load_and_preprocess_dataset(number_of_features):
    warnings.filterwarnings("ignore")
    df = load_dataset()
    preprocessed_features, preprocessed_label = preprocess_dataset(df, number_of_features)
    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Dataset loaded and preprocessed.")
    return preprocessed_features, preprocessed_label


def load_dataset():
    df = pd.read_csv('data/listings_Munich_July.csv')
    return df


def preprocess_dataset(df, number_of_features):

    # Select only the relevant features
    df = feature_selection(df)
    df = data_transformation(df)
    df = instance_selection(df)
    df = feature_extraction(df)
    df = noise_identification(df)
    df = data_cleaning(df)
    df = data_normalization(df)

    # Restore indices
    df = df.reset_index()
    df = df.drop('index', 1)



    label = df['max_price']

    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    best_features = selected_best_features(df, number_of_features)
    features = df[best_features]

    return features, label


def selected_best_features(df, number_of_features):
    # Split features and labels
    features = df.drop('max_price', 1)
    labels = df['max_price']

    # Select k best features using SelectKBest (k = 5)
    best_features = SelectKBest(score_func=chi2, k=number_of_features)
    fit = best_features.fit(features, labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(features.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    return featureScores['Specs']


def instance_selection(df):
    # Delete commercial listings of property_type
    #property_types = df['property_type']
    #commercial_property_types = ['Boutique hotel', 'Hostel', 'Hotel', 'Aparthotel']
    #commercial_indices = []
    #for index, value in property_types.iteritems():
    #    if value in commercial_property_types:
    #        commercial_indices.append(index)
    #df = df.drop(commercial_indices)

    #Delete special/unique listings (< 10 listings)
    #special_property_types = ['Guesthouse', 'Villa', 'Hut', 'Tiny house', 'Casa particular (Cuba)', 'Earth house', 'Tipi', 'Bus', 'Castle', 'Cave', 'Farm stay', 'Resort', 'Cabin', 'Boat', 'Yurt']
    #special_indices = []
    #for index, value in property_types.iteritems():
    #   if value in special_property_types:
    #       special_indices.append(index)
    #df = df.drop(special_indices)
    return df


def noise_identification(df):
    # Calculate mean and standard deviation
    price = df['max_price']
    price_mean = np.mean(price)
    price_std = np.std(price)

    # Detect noise and store their indices
    noise_indices = []
    for index, value in price.iteritems():
        if value > (price_mean + (3 * price_std)):
            noise_indices.append(index)
        if value < (price_mean - (3 * price_std)):
            noise_indices.append(index)

    # Delete noise
    df = df.drop(noise_indices)
    return df


def data_transformation(df):
    # Convert 'price' to float & Delete leading dollar sign
    df['price'] = df['price'].replace('[\$,)]', '', regex=True).astype(float)

    # Convert 'extra_people' to float & Delete leading dollar sign
    df['extra_people'] = df['extra_people'].replace('[\$,)]', '', regex=True).astype(float)
    return df


def feature_extraction(df):
    # Calculate the maximum price for all listings
    for index, row in df.iterrows():
        if row.guests_included < row.accommodates:
            df.loc[index, 'max_price'] = row.price + (row.accommodates - row.guests_included) * row.extra_people
            df.loc[index, 'max_people'] = row.accommodates - row.guests_included
        else:
            df.loc[index, 'max_price'] = row.price
            df.loc[index, 'max_people'] = row.accommodates + row.guests_included
    df = df.drop('guests_included', 1)
    df = df.drop('extra_people', 1)
    df = df.drop('price', 1)


    # Calculate deviation from city center
    df = geodata(df)
    return df


# Preprocess the feature 'neighbourhood'
def preprocess_neighbourhood(df):

    # Get list of distinct values of neighbourhoods
    neighbourhoods_selected = df.neighbourhood_cleansed.unique()

    # Create dummy variables for the neighbourhoods
    for element in neighbourhoods_selected:
        df[element.lower()] = 0

    # Set relevant values equal to 1
    for row in df.itertuples():
        for element in neighbourhoods_selected:
            if element in row.neighbourhood_cleansed:
                df.loc[row.Index, element.lower()] = 1

    # replace zero bedrooms with one bedroom
    df.loc[df.bedrooms == 0.0, 'bedrooms'] = 1.0
    return df


def data_normalization(df):
    # encode property_type
    # Alternative 1: Label encoding of property type
    label_encoder = preprocessing.LabelEncoder()
    # df['property_type'] = label_encoder.fit_transform(df['property_type'])

    # Alternative 2: Ordinal encoding of property type
    #ordinal_encoder = OrdinalEncoder()
    #order = ordinal_encoder.fit_transform(df[['property_type', 'max_price']].groupby('property_type').mean())
    #property_types = df['property_type'].unique()
    #for index, property_type in enumerate(property_types):
    #   df.loc[df.property_type == property_type, 'property_type'] = order[index][0]

    # encode room_type
    # Alternative 1: Label encoding of room type
    label_encoder = preprocessing.LabelEncoder()
    # df['room_type'] = label_encoder.fit_transform(df['room_type'])

    # Alternative 2: Ordinal encoding of property type
    ordinal_encoder = OrdinalEncoder()
    order = ordinal_encoder.fit_transform(df[['room_type', 'max_price']].groupby('room_type').mean())
    room_types = df['room_type'].unique()
    for index, room_type in enumerate(room_types):
        df.loc[df.room_type == room_type, 'room_type'] = order[index][0]

    #df['neighbourhood_cleansed'] = label_encoder.fit_transform(df['neighbourhood_cleansed'])

    # encode bathrooms
    # create replace map
    replace_map = {'bed_type': {'Couch': 1, 'Futon': 1, 'Pull-out Sofa': 1, 'Airbed': 2, 'Real Bed': 3}}

    # replace categorical values using replace map
    df = df.replace(replace_map)
    return df


def data_cleaning(df):
    # replace blank entries with shared bathroom
    #df.loc[df.bathrooms == None, 'bathrooms'] = 0.5
    df = df[~df['beds'].isnull()]
    #df = df[~df['review_scores_rating'].isnull()]
    #df = df[~df['bathrooms'].isnull()]
    df = df[~df['bedrooms'].isnull()]
    #df = df[~df['host_is_superhost'].isnull()]
    #df = df[~df['host_has_profile_pic'].isnull()]
    #df.loc[df.beds == "nan", 'beds'] = 0.5

    # replace zero bathrooms with shared bathroom
    #df.loc[df.bathrooms == 0.0, 'bathrooms'] = 0.5
    return df


def geodata(df):
    # Create point for every listing
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    crs = {'init': 'epsg:4326'}

    # create Geo-DataFrame with Points and Prices for each listing
    geo_df = gpd.GeoDataFrame(df['max_price'], crs=crs, geometry=geometry)

    # compute distance to Marienplatz as a proxy for the city centre
    geo_df['distance_centre'] = geo_df.distance(Point(11.576006, 48.137079))
    df['distance_centre'] = geo_df['distance_centre']
    df = df.drop('longitude', 1)
    df = df.drop('latitude', 1)

    # discretise distance_centre
    discretiser = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal')
    df['distance_centre'] = discretiser.fit_transform(df['distance_centre'].values.reshape(-1, 1))
    return df
