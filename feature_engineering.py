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
    # Create a new feature for total square footage
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    # Total number of bathrooms
    data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

    # Age of the house at the time of sale
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    # Age of the garage at the time of sale
    data['GarageAge'] = data['YrSold'] - data['GarageYrBlt']

    # Total porch area
    data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

    # Ratio of living area to lot area
    data['LivLotRatio'] = data['GrLivArea'] / data['LotArea']

    # Interaction between quality and condition
    data['QualCondInteraction'] = data['OverallQual'] * data['OverallCond']

    # Total number of rooms
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath']

    # Total square footage of the house including basement, garage, and porch
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
