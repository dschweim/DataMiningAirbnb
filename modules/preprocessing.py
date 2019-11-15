import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import geopandas as gpd
from shapely.geometry import Point


def load_and_preprocess_dataset():
    df = load_dataset()
    preprocessed_df = preprocess_dataset(df)
    return preprocessed_df


# Load the dataset containing airbnb listings in munich
def load_dataset():
    df = pd.read_csv('data/listings_Munich_July.csv')
    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Dataset loaded successfully.")
    return df


# Preprocess the dataset containing airbnb listings in munich
def preprocess_dataset(df):
    df = feature_selection(df)
    df = data_transformation(df)
    df = instance_selection(df)
    df = feature_extraction(df)
    df = noise_identification(df)
    df = data_cleaning(df)
    df = data_normalization(df)


    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Dataset preprocessed successfully.")
    return df


def feature_selection(df):
    selected_features = df.loc[:, df.columns.intersection(['accommodates',
                                            'amenities',
                                            'bathrooms',
                                            'bedrooms',
                                            'beds',
                                            'bed_type',
                                            'extra_people',
                                            'guests_included',
                                            'latitude',
                                            'longitude',
                                            'neighbourhood_cleansed',
                                            'property_type',
                                            'room_type',
                                            'price'])]
    return selected_features


def instance_selection(df):
    # Delete commercial listings of property_type
    property_types = df['property_type']
    commercial_property_types = ['Boutique hotel', 'Hostel', 'Hotel', 'Aparthotel']
    commercial_indices = []
    for index, value in property_types.iteritems():
        if value in commercial_property_types:
            commercial_indices.append(index)
    df = df.drop(commercial_indices)

    # Delete special/unique listings (< 10 listings)
    # special_property_types = ['Guesthouse', 'Villa', 'Hut', 'Tiny house', 'Casa particular (Cuba)', 'Earth house', 'Tipi', 'Bus', 'Castle', 'Cave', 'Farm stay', 'Resort', 'Cabin', 'Boat', 'Yurt']
    # special_indices = []
    # for index, value in property_types.iteritems():
    #    if value in special_property_types:
    #        special_indices.append(index)
    # munich = munich.drop(special_indices)
    # print(str(len(special_indices)), " special outliers deleted.")
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
    # label_encoder = preprocessing.LabelEncoder()
    # df['property_type'] = label_encoder.fit_transform(df['property_type'])

    # Alternative 2: Ordinal encoding of property type
    ordinal_encoder = OrdinalEncoder()
    order = ordinal_encoder.fit_transform(df[['property_type', 'max_price']].groupby('property_type').mean())
    property_types = df['property_type'].unique()
    for index, property_type in enumerate(property_types):
        df.loc[df.property_type == property_type, 'property_type'] = order[index][0]

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


    # encode bathrooms
    # create replace map
    replace_map = {'bed_type': {'Couch': 1, 'Futon': 1, 'Pull-out Sofa': 1, 'Airbed': 2, 'Real Bed': 3}}

    # replace categorical values using replace map
    df = df.replace(replace_map)
    return df


def data_cleaning(df):
    # replace blank entries with shared bathroom
    df.loc[df.bathrooms == None, 'bathrooms'] = 0.5
    df.loc[df.bathrooms == 'NaN', 'bathrooms'] = 0.5

    # replace zero bathrooms with shared bathroom
    df.loc[df.bathrooms == 0.0, 'bathrooms'] = 0.5
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

    # discretise distance_centre
    discretiser = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal')
    df['distance_centre'] = discretiser.fit_transform(df['distance_centre'].values.reshape(-1, 1))
    return df
