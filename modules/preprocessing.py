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
from collections import Counter
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest, SelectFwe
from sklearn.model_selection import train_test_split


# Preprocess the dataset
def load_and_preprocess_dataset():
    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Load the raw data set
    df = load_dataset()

    # Select and preprocess the relevant features
    reduced_df = feature_selection(df)

    # Preprocess the selected features
    preprocessed_df = preprocess_dataset(reduced_df)

    # Reset indices in the data set
    preprocessed_df = preprocessed_df.reset_index()
    preprocessed_df = preprocessed_df.drop('index', 1)

    # Split features and label
    features = preprocessed_df.drop(columns='maximum_price')
    label = preprocessed_df['maximum_price']

    print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), ": Dataset loaded and preprocessed.")
    return features, label


def load_dataset():
    df = pd.read_csv('data/listings_Munich_July.csv')
    return df


def feature_selection(df):
    selected_features = ['accommodates', 'amenities', 'bathrooms', 'bed_type', 'bedrooms', 'beds',
                         'cancellation_policy', 'cleaning_fee', 'extra_people',
                         'guests_included', 'host_has_profile_pic', 'host_identity_verified',
                         'host_is_superhost', 'host_location', 'host_total_listings_count',
                         'host_verifications', 'instant_bookable', 'is_location_exact', 'latitude', 'longitude',
                         'maximum_nights', 'minimum_nights', 'neighbourhood_cleansed', 'price', 'property_type',
                         'require_guest_phone_verification', 'require_guest_profile_picture', 'room_type',
                         'security_deposit']
    reduced_df = df.loc[:, df.columns.intersection(selected_features)]
    return reduced_df


def preprocess_dataset(df):
    # Preprocess label
    df = preprocess_price(df)

    # Preprocess features
    df = preprocess_amenities(df)
    df = preprocess_bathrooms(df)
    df = preprocess_bed_type(df)
    df = preprocess_bedrooms(df)
    df = preprocess_beds(df)
    df = preprocess_cancellation_policy(df)
    df = preprocess_cleaning_fee(df)
    df = preprocess_host_has_profile_pic(df)
    df = preprocess_host_identity_verified(df)
    df = preprocess_host_is_superhost(df)
    df = preprocess_host_listings_count(df)
    df = preprocess_host_location(df)
    df = preprocess_host_verifications(df)
    df = preprocess_instant_bookable(df)
    df = preprocess_is_location_exact(df)
    df = preprocess_maximum_nights(df)
    df = preprocess_minimum_nights(df)
    df = preprocess_neighbourhood_cleansed(df)
    df = preprocess_property_type(df)
    df = preprocess_require_guest_phone_verification(df)
    df = preprocess_require_guest_profile_picture(df)
    df = preprocess_room_type(df)
    df = preprocess_security_deposit(df)

    # Generate additional features
    df = generate_distance_to_city_center(df)
    df = generate_average_rent(df)
    return df


# Select the features with the highest correlation
def select_best_features(features, label, number_of_features):
    # Select k best features using SelectKBest (k = 5)
    best_features = SelectKBest(score_func=chi2, k=number_of_features)
    fit = best_features.fit(features, label)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(features.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    best_features = featureScores.nlargest(number_of_features, 'Score')['Specs']
    print("Selected Features: ", best_features)
    return features[best_features]


def select_best_features_f(features, label, number_of_features):
    f_value, p_value = f_regression(features, label)
    stat = pd.DataFrame({'feature': features.columns, 'f_value': f_value})
    stat = stat.sort_values('f_value', ascending=False)
    best_features = stat.head(number_of_features)['feature']
    return features[best_features]


def stratified_train_test_split(features, label):
    bins = pd.qcut(label, 30, labels=False)
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42, stratify=bins)


# Preprocess the features in the dataset
def preprocess_amenities(df):
    # Create list of all individual amenities
    all_amenities = list()
    for amenities in df['amenities']:
        for amenity in amenities.split("{")[1].split("}")[0].split(","):
            all_amenities.append(amenity.replace('"', ''))

    # all_amenities_set = set(all_amenities)
    Counter(all_amenities).most_common(50)
    relevant_amenities = ['Wifi', 'Internet', 'TV', 'Kitchen', 'Heating', 'Washer', 'Patio or balcony',
                          'Breakfast', 'Elevator', '24-hour check-in', 'Pool',
                          'Private entrance', 'Dishwasher', 'Bed linens', 'Smoking allowed']
    for element in relevant_amenities:  # create dummy variables
        df[element.lower()] = 0
    for row in df.itertuples():  # can take a moment
        for element in relevant_amenities:
            if element in row.amenities:
                df.loc[row.Index, element.lower()] = 1
    df = df.drop(columns='amenities')

    # merge information from variables wifi and internet in the feature 'internet'
    for row in df.itertuples():
        if row.wifi == 1:
            df.loc[row.Index, 'internet'] = 1
    df = df.drop(columns=['wifi', 'pool', 'bed linens'])
    df['internet'].describe()
    return df


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
    df.loc[df.bedrooms.isna(), 'bedrooms'] = 0.0
    return df


def preprocess_beds(df):
    df.loc[df.beds.isna(), 'beds'] = 0.0
    return df


def preprocess_cancellation_policy(df):
    df.loc[df.cancellation_policy == 'super_strict_60', 'cancellation_policy'] = 4
    df.loc[df.cancellation_policy == 'strict_14_with_grace_period', 'cancellation_policy'] = 3
    df.loc[df.cancellation_policy == 'strict', 'cancellation_policy'] = 2
    df.loc[df.cancellation_policy == 'moderate', 'cancellation_policy'] = 1
    df.loc[df.cancellation_policy == 'flexible', 'cancellation_policy'] = 0
    return df


def preprocess_cleaning_fee(df):
    # Convert 'cleaning_fee' to float & Delete leading dollar sign
    df['cleaning_fee'] = df['cleaning_fee'].replace('[\$,)]', '', regex=True).astype(float)

    # Replace blank entries with 0
    df.loc[df.cleaning_fee.isna(), 'cleaning_fee'] = 0.0
    return df


def preprocess_extra_people(df):
    df['extra_people'] = df['extra_people'].replace('[\$,)]', '', regex=True).astype(float)
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


def preprocess_host_listings_count(df):
    df.loc[df.host_total_listings_count.isna(), 'host_total_listings_count'] = 0.0
    return df


def preprocess_host_location(df):
    df['host_lives_in_munich'] = 0
    df['host_location'] = df['host_location'].astype(str)
    munich_name = ['Munich', 'München']
    for index, row in df.iterrows():
        for element in munich_name:
            if element in row.host_location:
                df.loc[index, 'host_lives_in_munich'] = 1
    df = df.drop(columns='host_location')
    return df


def preprocess_host_verifications(df):
    all_verifications = list()  # create list of all individual verifications
    for verifications in df[df.notnull()]['host_verifications']:
        if "," in verifications:
            for element in verifications.split("[")[1].split("]")[0].split(","):
                all_verifications.append(element.replace("'", ""))
        else:
            all_verifications.append(element.replace("'", ""))
    Counter(all_verifications).most_common(50)
    relevant_verifications = ['email', 'phone', 'reviews', 'government_id', 'jumio', 'offline_government_id',
                              'facebook', 'selfie', 'work_email', 'google']
    for element in relevant_verifications:  # create dummy variables
        df[f"verification_{element}"] = 0
    for row in df.itertuples():  # can take a moment
        for element in relevant_verifications:
            if element in row.host_verifications:
                df.loc[row.Index, f"verification_{element}"] = 1
    df = df.drop(columns='host_verifications')
    return df


def preprocess_instant_bookable(df):
    df = df[~df['instant_bookable'].isnull()]
    label_encoder = preprocessing.LabelEncoder()
    df['instant_bookable'] = label_encoder.fit_transform(df['instant_bookable'])
    return df


def preprocess_is_location_exact(df):
    df.loc[df.is_location_exact == 't', 'is_location_exact'] = 1.0
    df.loc[df.is_location_exact == 'f', 'is_location_exact'] = 0.0
    return df


def preprocess_maximum_nights(df):
    return df


def preprocess_minimum_nights(df):
    # Create bin for short-term bookings
    df.loc[df.minimum_nights <= 3, 'minimum_nights'] = 0

    # Create bin for long-term bookings
    df.loc[((df.minimum_nights > 3) & (df.minimum_nights <= 15)), 'minimum_nights'] = 1

    # Create bin for holiday bookings
    df.loc[df.minimum_nights > 15, 'minimum_nights'] = 2
    return df


def preprocess_neighbourhood_cleansed(df):
    # Get list of distinct values of neighbourhoods
    neighbourhoods_selected = df.neighbourhood_cleansed.unique()

    # Create dummy variables for the neighbourhoods
    for element in neighbourhoods_selected:
        df[element.lower()] = 0

    # Set relevant values equal to 1
    for row in df.itertuples():
        for element in neighbourhoods_selected:
            if element == row.neighbourhood_cleansed:
                df.loc[row.Index, element.lower()] = 1

    return df


def preprocess_price(df):
    # Convert 'price' to float & Delete leading dollar sign
    df['price'] = df['price'].replace('[\$,)]', '', regex=True).astype(float)

    # Convert 'extra_people' to float & Delete leading dollar sign
    df = preprocess_extra_people(df)

    # Calculate the maximum price for all listings
    df = generate_maximum_listing_price(df)
    # Drop the features 'price' because of new feature 'maximum_price'
    df = df.drop(['price', 'guests_included', 'extra_people'], 1)
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
    # label_encoder = preprocessing.LabelEncoder()
    # df['property_type'] = label_encoder.fit_transform(df['property_type'])

    # Alternative 2: Ordinal encoding of property type
    ordinal_encoder = OrdinalEncoder()
    order = ordinal_encoder.fit_transform(df[['property_type', 'maximum_price']].groupby('property_type').mean())
    property_types = df['property_type'].unique()
    for index, property_type in enumerate(property_types):
       df.loc[df.property_type == property_type, 'property_type'] = order[index][0]
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
    # Alternative 1: Label encoding of room type
    # label_encoder = preprocessing.LabelEncoder()
    # df['room_type'] = label_encoder.fit_transform(df['room_type'])

    # Alternative 2: Ordinal encoding of property type
    ordinal_encoder = OrdinalEncoder()
    order = ordinal_encoder.fit_transform(df[['room_type', 'maximum_price']].groupby('room_type').mean())
    room_types = df['room_type'].unique()
    for index, room_type in enumerate(room_types):
        df.loc[df.room_type == room_type, 'room_type'] = order[index][0]
    return df


def preprocess_security_deposit(df):
    # Convert 'security_deposit' to float & Delete leading dollar sign
    df['security_deposit'] = df['security_deposit'].replace('[\$,)]', '', regex=True).astype(float)

    # Replace blank entries with 0
    df.loc[df.security_deposit.isna(), 'security_deposit'] = 0.0
    return df


# Generate new features
def generate_maximum_listing_price(df):
    for index, row in df.iterrows():
        if row.guests_included < row.accommodates:
            df.loc[index, 'maximum_price'] = row.price + (row.accommodates - row.guests_included) * row.extra_people
        else:
            df.loc[index, 'maximum_price'] = row.price
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


def generate_average_rent(df):
    rent_df = pd.read_csv('data/rent_index_Munich.csv')
    df['average_rent_neighbourhood'] = 0
    for index, row in df.iterrows():
        for r_index, r_row in rent_df.iterrows():
            if row.neighbourhood_cleansed == r_row.neighbourhood:
                df.loc[index, 'average_rent_neighbourhood'] = r_row.average_rent_euro * 1.1

    # Delete initial column neighbourhood_cleansed
    df = df.drop(columns='neighbourhood_cleansed')

    return df


def delete_price_outliers(df_x, df_y):
    # Calculate mean and standard deviation
    price_mean = np.mean(df_y)
    price_std = np.std(df_y)

    # Detect noise and store their indices
    noise_indices = []
    for index, value in df_y.iteritems():
        if value > (price_mean + (3 * price_std)):
            noise_indices.append(index)
        if value < (price_mean - (3 * price_std)):
            noise_indices.append(index)

    # Delete noise
    df_y = df_y.drop(noise_indices)
    df_x = df_x.drop(noise_indices)
    return df_x, df_y