# 1. Import necessary libraries.
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


# 2. Display a sample of five rows of the data frame.
df = pd.read_csv('sales_data_sample.csv')
print(df.head(5))

# 3. Check the shape of the data (number of rows and columns). Check the general information about the dataframe using the .info() method.
print("Shape:", df.shape)
print(df.info())

# 4. Check the percentage of missing values in each column of the data frame.
missing_percentage = df.isnull().sum() / len(df) * 100
print("Missing values percentage:")
print(missing_percentage)

# 5. Check if there are any duplicate rows.
duplicate_rows = df.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())

# 6. Write a function that will impute missing values of the columns “STATE”, “POSTALCODE”,“TERRITORY” with its most occurring label.
def impute_missing_values(df):
    df['STATE'].fillna(df['STATE'].mode()[0], inplace=True)
    df['POSTALCODE'].fillna(df['POSTALCODE'].mode()[0], inplace=True)
    df['TERRITORY'].fillna(df['TERRITORY'].mode()[0], inplace=True)

impute_missing_values(df)

# 7. Drop “ADDRESSLINE2”,”ORDERDATE”,”PHONE” column.
df.drop(['ADDRESSLINE2', 'ORDERDATE', 'PHONE'], axis=1, inplace=True)

# 8. Convert the labels of the STATUS column to 0 and 1. For Shipped assign value 1 and for all other labels assign 0.
label_encoder = LabelEncoder()
df['STATUS'] = label_encoder.fit_transform(df['STATUS'])
df['STATUS'] = np.where(df['STATUS'] == 1, 1, 0)

print(df.head())
