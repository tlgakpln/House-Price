import pandas as pd
from sklearn.impute import SimpleImputer
import feature_engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from config import configs


def handle_missing_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(data)
    data_filled = pd.DataFrame(data_filled, columns=data.columns)
    return data_filled


def one_hot_encode(data, categorical_columns):
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = encoder.fit_transform(data[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    data = pd.concat([data, encoded_df], axis=1)
    data.drop(categorical_columns, axis=1, inplace=True)
    return data


def scale_data(data, numeric_columns):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
    data = pd.concat([data.drop(numeric_columns, axis=1), scaled_df], axis=1)
    return data


def main():
    df = feature_engineering.main_program()
    df = one_hot_encode(df, configs.categorical_columns)
    df = handle_missing_data(df)
    df = scale_data(df, configs.numerical_columns)


    print("Processed data:\n", df.head())
    return df


if __name__ == "__main__":
    main()
