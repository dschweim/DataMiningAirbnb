import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OrdinalEncoder


def load_and_preprocess_dataset():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Load the raw data set
    df = load_dataset()

    # Select and preprocess the relevant features
    features = feature_selection(df)

    # Preprocess the selected features
    preprocessed_features = preprocess_dataset(features)
    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    # best_features = selected_best_features(df, number_of_features)
    # features = df[best_features]

    # Reset indices in the data set
    preprocessed_features = preprocessed_features.reset_index()
    preprocessed_features = preprocessed_features.drop('index', 1)

    # Split and preprocess the label
    label = preprocessed_features['maximum_price']
    features = preprocessed_features.drop(['maximum_price', 'price'], 1)
    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Dataset loaded and preprocessed.")
    return label, features


def load_dataset():
    df = pd.read_csv('data/listings_Munich_July.csv')
    return df


def preprocess_dataset(df):
    # Preprocess label
    df = preprocess_price(df)

    # Preprocess features
    df = preprocess_bathrooms(df)
    df = preprocess_bed_type(df)
    df = preprocess_bedrooms(df)
    df = preprocess_beds(df)
    df = preprocess_has_availability(df)
    df = preprocess_host_has_profile_pic(df)
    df = preprocess_host_identity_verified(df)
    df = preprocess_host_is_superhost(df)
    df = preprocess_host_location(df)
    df = preprocess_instant_bookable(df)
    df = preprocess_neighbourhood_cleansed(df)
    df = preprocess_property_type(df)
    df = preprocess_require_guest_phone_verification(df)
    df = preprocess_require_guest_profile_picture(df)
    df = preprocess_room_type(df)

    # Generate additional features
    df = generate_distance_to_city_center(df)
    return df


def feature_selection(df):
    selected_features = ['accommodates', 'amenities', 'availability_30', 'availability_365', 'availability_60',
                         'availability_90', 'bathrooms', 'bed_type', 'bedrooms', 'beds',
                         'calculated_host_listings_count', 'cancellation_policy', 'cleaning_fee', 'extra_people',
                         'guests_included', 'has_availability', 'host_has_profile_pic', 'host_identity_verified',
                         'host_is_superhost', 'host_location', 'host_since', 'host_total_listings_count',
                         'host_verifications', 'instant_bookable', 'is_location_exact', 'latitude', 'longitude',
                         'maximum_nights', 'minimum_nights', 'neighbourhood_cleansed', 'price', 'property_type',
                         'require_guest_phone_verification', 'require_guest_profile_picture', 'room_type',
                         'security_deposit', 'zipcode']
    reduced_df = df.loc[:, df.columns.intersection(selected_features)]
    return reduced_df


def selected_best_features(df, number_of_features):
    # Split features and labels
    features = df.drop('maximum_price', 1)
    labels = df['maximum_price']

    # Select k best features using SelectKBest (k = 5)
    best_features = SelectKBest(score_func=chi2, k=number_of_features)
    fit = best_features.fit(features, labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(features.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    return featureScores['Specs']


# Preprocess the features in the dataset
def preprocess_bathrooms(df):
    df = df[~df['bathrooms'].isnull()]
    df.loc[df.bathrooms == None, 'bathrooms'] = 0.5
    df.loc[df.bathrooms == 0.0, 'bathrooms'] = 0.5
    return df


def preprocess_bed_type(df):
    replace_map = {'bed_type': {'Couch': 1, 'Futon': 1, 'Pull-out Sofa': 1, 'Airbed': 2, 'Real Bed': 3}}
    df = df.replace(replace_map)
    return df


def preprocess_bedrooms(df):
    df = df[~df['bedrooms'].isnull()]
    df.loc[df.bedrooms == 0.0, 'bedrooms'] = 1.0
    return df


def preprocess_beds(df):
    df = df[~df['beds'].isnull()]
    df.loc[df.beds == "nan", 'beds'] = 0.5
    return df


def preprocess_extra_people(df):
    df['extra_people'] = df['extra_people'].replace('[\$,)]', '', regex=True).astype(float)
    return df


def preprocess_has_availability(df):
    df = df[~df['has_availability'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['has_availability'] = label_encoder.fit_transform(df['has_availability'])
    return df


def preprocess_host_has_profile_pic(df):
    df = df[~df['host_has_profile_pic'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['host_has_profile_pic'] = label_encoder.fit_transform(df['host_has_profile_pic'])
    return df


def preprocess_host_identity_verified(df):
    df = df[~df['host_identity_verified'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['host_identity_verified'] = label_encoder.fit_transform(df['host_identity_verified'])
    return df


def preprocess_host_is_superhost(df):
    df = df[~df['host_is_superhost'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['host_is_superhost'] = label_encoder.fit_transform(df['host_is_superhost'])
    return df


def preprocess_host_location(df):
    return df

def preprocess_instant_bookable(df):
    df = df[~df['instant_bookable'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['instant_bookable'] = label_encoder.fit_transform(df['instant_bookable'])
    return df


def preprocess_minimum_nights(df):
    print(pd.cut(df['minimum_nights'], bins=6))
    print(df[['minimum_nights', 'maximum_price']].groupby('minimum_nights').mean())
    return df


def preprocess_neighbourhood_cleansed(df):
    label_encoder = preprocessing.LabelEncoder()
    df['neighbourhood_cleansed'] = label_encoder.fit_transform(df['neighbourhood_cleansed'])
    return df


def preprocess_price(df):
    # Convert 'price' to float & Delete leading dollar sign
    df['price'] = df['price'].replace('[\$,)]', '', regex=True).astype(float)

    # Convert 'extra_people' to float & Delete leading dollar sign
    df = preprocess_extra_people(df)

    # Calculate the maximum price for all listings
    df = generate_maximum_listing_price(df)

    # Calculate mean and standard deviation
    price = df['maximum_price']
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


def preprocess_property_type(df):
    # Delete commercial listings of property_type
    property_types = df['property_type']
    commercial_property_types = ['Boutique hotel', 'Hostel', 'Hotel', 'Aparthotel']
    commercial_indices = []
    for index, value in property_types.iteritems():
        if value in commercial_property_types:
            commercial_indices.append(index)
    df = df.drop(commercial_indices)


    # Alternative 1: Label encoding of property type
    label_encoder = preprocessing.LabelEncoder()
    df['property_type'] = label_encoder.fit_transform(df['property_type'])

    # Alternative 2: Ordinal encoding of property type
    #ordinal_encoder = OrdinalEncoder()
    #order = ordinal_encoder.fit_transform(df[['property_type', 'maximum_price']].groupby('property_type').mean())
    #property_types = df['property_type'].unique()
    #for index, property_type in enumerate(property_types):
    #   df.loc[df.property_type == property_type, 'property_type'] = order[index][0]
    return df


def preprocess_require_guest_phone_verification(df):
    df = df[~df['require_guest_phone_verification'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['require_guest_phone_verification'] = label_encoder.fit_transform(df['require_guest_phone_verification'])
    return df


def preprocess_require_guest_profile_picture(df):
    df = df[~df['require_guest_profile_picture'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['require_guest_profile_picture'] = label_encoder.fit_transform(df['require_guest_profile_picture'])
    return df


def preprocess_room_type(df):
    label_encoder = preprocessing.LabelEncoder()
    df['room_type'] = label_encoder.fit_transform(df['room_type'])

    # Alternative 2: Ordinal encoding of property type
    #ordinal_encoder = OrdinalEncoder()
    #order = ordinal_encoder.fit_transform(df[['room_type', 'maximum_price']].groupby('room_type').mean())
    #room_types = df['room_type'].unique()
    #for index, room_type in enumerate(room_types):
    #    df.loc[df.room_type == room_type, 'room_type'] = order[index][0]
    return df


# Generate new features
def generate_maximum_listing_price(df):
    for index, row in df.iterrows():
        if row.guests_included < row.accommodates:
            df.loc[index, 'maximum_price'] = row.price + (row.accommodates - row.guests_included) * row.extra_people
            df.loc[index, 'maximum_people'] = row.accommodates - row.guests_included
        else:
            df.loc[index, 'maximum_price'] = row.price
            df.loc[index, 'maximum_people'] = row.accommodates + row.guests_included
    return df


def generate_distance_to_city_center(df):
    # Create point for every listing
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    crs = {'init': 'epsg:4326'}

    # create Geo-DataFrame with Points and Prices for each listing
    geo_df = gpd.GeoDataFrame(df['maximum_price'], crs=crs, geometry=geometry)

    # compute distance to Marienplatz as a proxy for the city centre
    geo_df['distance_centre'] = geo_df.distance(Point(11.576006, 48.137079))
    df['distance_centre'] = geo_df['distance_centre']
    df = df.drop('longitude', 1)
    df = df.drop('latitude', 1)

    # discretise distance_centre
    discretiser = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal')
    df['distance_centre'] = discretiser.fit_transform(df['distance_centre'].values.reshape(-1, 1))
    return df

