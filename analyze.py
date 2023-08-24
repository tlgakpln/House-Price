import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def analyze_missing_data(data):
    missing_data = data.isnull().sum()
    missing_data_percentage = (missing_data / len(data)) * 100
    missing_info = pd.concat([missing_data, missing_data_percentage], axis=1, keys=['Missing Data Count', 'Missing Data Percentage'])
    missing_info = missing_info[missing_info['Missing Data Count'] > 0]
    print("Missing Data Analysis:\n", missing_info)

def handle_missing_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(data)
    data_filled = pd.DataFrame(data_filled, columns=data.columns)
    return data_filled

def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    data['CategoricalColumn'] = label_encoder.fit_transform(data['CategoricalColumn'])  # Example column name
    return data

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

def visualize_histogram(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data['SalePrice'], bins=30, color='blue', alpha=0.7)
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.title('Sale Price Histogram')
    plt.show()

def visualize_boxplot(data, column_name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column_name], y=data['SalePrice'])
    plt.xticks(rotation=90)
    plt.xlabel(column_name)
    plt.ylabel('Sale Price')
    plt.title(f'Sale Price by {column_name}')
    plt.show()

def visualize_scatterplot(data, x_column, y_column):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.7)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{x_column} vs {y_column}')
    plt.show()

def visualize_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Features')
    plt.show()

def visualize_pairplot(data, columns):
    sns.pairplot(data[columns])
    plt.suptitle('Pair Plot of Selected Features', y=1.02)
    plt.show()

def visualize_categorical_boxplot(data, categorical_column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[categorical_column], y=data['SalePrice'])
    plt.xticks(rotation=45)
    plt.xlabel(categorical_column)
    plt.ylabel('Sale Price')
    plt.title(f'Sale Price by {categorical_column}')
    plt.show()

def main_program():
    file_path = 'dfs/train.csv'
    data = load_data(file_path)

    analyze_missing_data(data)
    data = handle_missing_data(data)

    data = encode_categorical_data(data)

    data = normalize_data(data)

    visualize_histogram(data)
    visualize_boxplot(data, 'OverallQual')
    visualize_scatterplot(data, 'GrLivArea', 'SalePrice')
    visualize_correlation_matrix(data)
    visualize_pairplot(data, ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF'])
    visualize_categorical_boxplot(data, 'Neighborhood')

if __name__ == "__main__":
    main_program()
