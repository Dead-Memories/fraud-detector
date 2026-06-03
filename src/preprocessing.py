import pandas as pd
import numpy as np
import logging
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

TARGET_COL = 'target'
CATEGORICAL_COLS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
DROP_COLS = ['name_1', 'name_2', 'street', 'post_code']
CONTINUOUS_COLS = ['amount', 'population_city']
N_CATS = 50


def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df


def cat_encode(train, input_df, col):
    logger.debug('Encoding category: %s', col)
    new_col = col + '_cat'
    mapping = train[[col, new_col]].drop_duplicates()
    input_df = input_df.merge(mapping, how='left', on=col).drop(columns=col)
    return input_df


def add_distance_features(df):
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']),
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])


def load_train_data():
    logger.info('Loading training data...')

    train = pd.read_csv('./train_data/train.csv').drop(columns=DROP_COLS)
    logger.info('Raw train data imported. Shape: %s', train.shape)

    train = add_time_features(train)

    for col in CATEGORICAL_COLS:
        new_col = col + '_cat'
        temp_df = train\
            .groupby(col, dropna=False)[[TARGET_COL]]\
            .count()\
            .sort_values(TARGET_COL, ascending=False)\
            .reset_index()\
            .set_axis([col, 'count'], axis=1)\
            .reset_index()
        temp_df['index'] = temp_df.apply(lambda x: np.nan if pd.isna(x[col]) else x['index'], axis=1)
        temp_df[new_col] = ['cat_NAN' if pd.isna(x) else 'cat_' + str(x) if x < N_CATS else f'cat_{N_CATS}+' for x in temp_df['index']]
        train = train.merge(temp_df[[col, new_col]], how='left', on=col)

    train = add_distance_features(train)
    logger.info('Train data processed. Shape: %s', train.shape)
    return train


def run_preproc(train, input_df):
    input_df = input_df.drop(columns=[c for c in DROP_COLS if c in input_df.columns])

    for col in CATEGORICAL_COLS:
        input_df = cat_encode(train, input_df, col)
    logger.info('Categorical merging completed. Shape: %s', input_df.shape)

    input_df = add_time_features(input_df)
    logger.info('Time features added. Shape: %s', input_df.shape)

    categorical_encoded = [x + '_cat' for x in CATEGORICAL_COLS]
    time_cols = ['hour', 'year', 'month', 'day_of_month', 'day_of_week']
    categorical_encoded.extend(time_cols)

    for col in categorical_encoded:
        input_df[col] = input_df[col].fillna('cat_NAN')
        means_tb = train.groupby(col)[[TARGET_COL]].mean()\
                        .reset_index().rename(columns={TARGET_COL: f'{col}_mean_enc'})
        input_df = input_df.merge(means_tb, how='left', on=col)

    logger.info('Mean encoding completed. Shape: %s', input_df.shape)

    input_df = add_distance_features(input_df)
    all_continuous = CONTINUOUS_COLS + ['distance']

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(train[all_continuous])

    remaining_cols = [c for c in input_df.columns if c not in all_continuous + categorical_encoded]
    cont_transformed = pd.DataFrame(
        imputer.transform(input_df[all_continuous]),
        columns=all_continuous,
        index=input_df.index
    )
    output_df = pd.concat([input_df[remaining_cols], cont_transformed], axis=1)

    for col in all_continuous:
        output_df[col + '_log'] = np.log(output_df[col] + 1)
        output_df.drop(columns=col, inplace=True)

    drop_cols = [TARGET_COL] if TARGET_COL in output_df.columns else []
    output_df = output_df.drop(columns=drop_cols)

    logger.info('Preprocessing complete. Shape: %s', output_df.shape)
    return output_df
