import analyze
import pandas as pd
from config import configs


def edit_dtypes(data):
        # Convert categorical columns to categorical data type
    for col in configs.categorical_columns:
        data[col] = data[col].astype('category')

    # Convert numerical columns to appropriate numerical data types
    for col in configs.numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def engineer_new_features(data):
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    data['GarageAge'] = data['YrSold'] - data['GarageYrBlt']

    data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

    data['LivLotRatio'] = data['GrLivArea'] / data['LotArea']

    data['QualCondInteraction'] = data['OverallQual'] * data['OverallCond']

    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath']

    data['TotalHouseSF'] = data['TotalSF'] + data['TotalPorchSF']

    return data

def main_program():
    file_path = 'dfs/train.csv'
    data = analyze.load_data(file_path)
    data = edit_dtypes(data)
    data = engineer_new_features(data)
    return data



if __name__ == "__main__":
    main_program()
