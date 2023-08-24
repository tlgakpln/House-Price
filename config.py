class configs:
    numerical_columns = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                         'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                         'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                         'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                         'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                         'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                         'MoSold', 'YrSold', 'SalePrice']

    categorical_columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                           'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                           'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                           'SaleCondition']