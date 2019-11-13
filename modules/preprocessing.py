import numpy as np
import pandas as pd
import warnings

from sklearn import preprocessing

# Ignore all warnings
warnings.filterwarnings("ignore")


# Load the dataset containing airbnb listings in munich
def load_dataset():
    # Load dataset
    df = pd.read_csv('data/listings_Munich_July.csv')
    print("Dataset loaded successfully.")
    return df


# Preprocess the dataset
def load_and_preprocess_dataset():
    # Load dataset
    df = load_dataset()

    # Select only the relevant features
    df = df.loc[:, df.columns.intersection(['accommodates',
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
    print("- Dropped all irrelevant features.")

    # Preprocess the price.
    df = preprocess_price(df)

    # One-hot encoding for neighbourhoods

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

    # ordinal encoding
    # create replace map
    replace_map = {'bed_type': {'Couch': 1, 'Futon': 1, 'Pull-out Sofa': 1, 'Airbed': 2, 'Real Bed': 3}}

    # replace categorical values using replace map
    df = df.replace(replace_map)

    # replace blank entries with shared bathroom
    df.loc[df.bathrooms == None, 'bathrooms'] = 0.5

    # replace zero bathrooms with shared bathroom
    df.loc[df.bathrooms == 0.0, 'bathrooms'] = 0.5

    # Note: There are no missing values for both property_type and room_type. The formal check will added to the code later on anyway.
    from sklearn.preprocessing import OrdinalEncoder

    # Alternative 1: Label encoding of property type
    label_encoder = preprocessing.LabelEncoder()
    df['property_type'] = label_encoder.fit_transform(df['property_type'])

    # Alternative 2: Ordinal encoding of property type
    # ordinal_encoder = OrdinalEncoder()
    # order = ordinal_encoder.fit_transform(munich[['property_type', 'price']].groupby('property_type').mean())
    # property_types = munich['property_type'].unique()
    # for index, property_type in enumerate(property_types):
    #     munich[munich.property_type == property_type].property_type = order[index][0]

    # Note: There are no missing values for both property_type and room_type. The formal check will added to the code later on anyway.
    from sklearn.preprocessing import OrdinalEncoder

    # Alternative 1: Label encoding of room type
    label_encoder = preprocessing.LabelEncoder()
    df['room_type'] = label_encoder.fit_transform(df['room_type'])

    # Alternative 2: Ordinal encoding of property type
    # ordinal_encoder = OrdinalEncoder()
    # order = ordinal_encoder.fit_transform(munich[['room_type', 'price']].groupby('room_type').mean())
    # room_types = munich['room_type'].unique()
    # for index, room_type in enumerate(room_types):
    #     munich[munich.room_type == room_type].room_type = order[index][0]

    # drop unused features
    df = df.drop(columns=['neighbourhood_cleansed', 'property_type', 'room_type', 'price'])

    delete_outlier(df)
    print("Dataset preprocessed successfully.")
    return df


# Preprocess the price
def preprocess_price(df):
    # Cast 'price' to float & Delete leading dollar sign
    df['price'] = df['price'].replace('[\$,)]', '', regex=True).astype(float)

    # Cast 'extra_people' to float & Delete leading dollar sign
    df['extra_people'] = df['extra_people'].replace('[\$,)]', '', regex=True).astype(float)

    # Calculate the maximum price for all listings
    for index, row in df.iterrows():
        if row.guests_included < row.accommodates:
            df.loc[index, 'max_price'] = row.price + (row.accommodates - row.guests_included) * row.extra_people
        else:
            df.loc[index, 'max_price'] = row.price

    # Drop now obsolete features
    df = df.drop(columns=['extra_people', 'guests_included', 'price'])
    print("- Price preprocessed.")
    return df


# Outlier detection and deletion via standard deviation
def delete_outlier(df):
    # Calculate mean and standard deviation
    price = df['max_price']
    price_mean = np.mean(price)
    price_std = np.std(price)

    # Detect outliers and store their indices
    outlier_indices = []
    for index, value in price.iteritems():
        if value > (price_mean + (3 * price_std)):
            outlier_indices.append(index)

    # Delete outliers
    df = df.drop(outlier_indices)
    print("- ", str(len(outlier_indices)), " outliers deleted.")
    return df

#def geodata():
#    # read shapefile of Bavaria\n",
#    munich_sh = 'Data//Geodata/gmd_ex.shp'
#    map_df = gpd.read_file(munich_sh)
#
#    # Create point for every listing
#    geometry = [Point(xy) for xy in zip(munich['longitude'], munich['latitude'])]
#    munich = munich.drop(columns=['longitude', 'latitude'])
#    crs = {
#        'init': 'epsg:4326'}  # crs-data tells geopandas how the data relate to points in the real worlds (e.g. latitude and longitude)
#    map_df = map_df.to_crs(epsg=4326)  # convert given shapefile to latitude and longitude-crs
#
#    # create Geo-DataFrame with Points and Prices for each listing
#    geo_df = gpd.GeoDataFrame(munich['max_price'], crs=crs, geometry=geometry)
#
#    # create DataFrame for visuals
#    geom = [Point(xy) for xy in zip(munich_visual['longitude'], munich_visual['latitude'])]
#    geo_df_visual = gpd.GeoDataFrame(munich_visual['max_price'], crs=crs, geometry=geom)
#
#    # compute distance to Marienplatz as a proxy for the city centre
#    geo_df['distance_centre'] = geo_df.distance(Point(11.576006, 48.137079))
#
#    # add feature to DataFrame and create a Geopandas-object for plotting
#    munich['distance_centre'] = geo_df['distance_centre']
#    point_marienplatz = gpd.GeoSeries(Point(11.576006, 48.137079), crs=crs)
#
# #discretise distance_centre
# discretiser = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal')
# munich['distance_centre'] = discretiser.fit_transform(munich['distance_centre'].values.reshape(-1, 1))