# CS 482 - Assignment 3
# Author: Demetri Karras
# File: karras3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

data = pd.read_csv("houseSalePrices.csv")

""" ******** Meet the Data ******** """

num_features = data.shape[1] - 1
print(f"Number of Features: {num_features}\n")
print(f"Names of Features:\n{list(data.columns[:-1])}\n")
print(f"Name of Target: {data.columns[-1]}\n")
num_samples = data.shape[0]
print(f"Number of Samples: {num_samples}\n")
print(f"First 5 Rows:\n{data.head()}\n")

""" ******** Split the Data ******** """

X = data.iloc[:, :-1]
y = data.iloc[:, -1]  # target column
x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    shuffle=True, 
                                                    random_state=42)

""" ******** Data Preprocessing ******** """

# drop singular identifier column
x_train = x_train.drop(columns=["Id"], axis=1)
x_test = x_test.drop(columns=["Id"], axis=1)

""" -- Missing Value Handling -- """

# obtain features that have missing values along with all their unique values
print("-- Features with NaN Values --\n")
missing_value_features = x_train.columns[x_train.isna().any()]
for mvf in missing_value_features:

    total_missing = x_train[mvf].isna().sum()
    percent_missing = round(x_train[mvf].isna().mean() * 100, 2) # quantity of NaN values for a feature multiplied by 100 and rounded to two decimals

    print(f"{mvf}: {total_missing} ({percent_missing}%)")
    print(f"Unique Values: {x_train[mvf].unique()}\n")

print("-- Masonry Veneer Metrics --\n")

# metrics for masonry veneers - need to take a closer look at related features after seeing missing value counts
print(f"NaN MasVnrType Total: {x_train['MasVnrType'].isna().sum()}")
print(f"NaN MasVnrArea Total: {((x_train['MasVnrArea'].isna()) | (x_train['MasVnrArea'] == 0)).sum()}\n")

# obtain lists and print metrics for masonry veneers that don't exist, and masonry veneers that were not fully recorded
true_missing_masonry_cond = x_train['MasVnrType'].isna() & (x_train['MasVnrArea'].isna() | (x_train['MasVnrArea'] == 0))
valid_type_nan_area_cond = ~(x_train['MasVnrType'].isna()) & (x_train['MasVnrArea'].isna() | (x_train['MasVnrArea'] == 0))
nan_type_valid_area_cond = x_train['MasVnrType'].isna() & ~(x_train['MasVnrArea'].isna() | (x_train['MasVnrArea'] == 0))
print(f"NaN MasVnrType + NaN MasVnrArea Count: {true_missing_masonry_cond.sum()}") # truly missing masonry veneers
print(f"Valid MasVnrType + NaN MasVnrArea Count: {valid_type_nan_area_cond.sum()}") # area not calculated
print(f"NaN MasVnrType + Valid MasVnrArea Count: {nan_type_valid_area_cond.sum()}\n") # type not recorded

x_train.loc[true_missing_masonry_cond, "MasVnrType"] = "None" # "None" assigned to masonry veneer types that don't exist
x_train.loc[true_missing_masonry_cond, "MasVnrArea"] = 0 # 0 assigned to masonry veneer areas that don't exist

# median obtained and assigned to masonry veneer areas that were not recorded
masvnrarea_median = x_train.loc[(x_train["MasVnrArea"].notna()) & (x_train["MasVnrArea"] > 0), "MasVnrArea"].median()
x_train.loc[valid_type_nan_area_cond, "MasVnrArea"] = masvnrarea_median

# mode obtained and assigned to masonry veneer types that were not recorded
masvnrtype_mode = x_train.loc[x_train["MasVnrType"].notna(), "MasVnrType"].mode()[0]
x_train.loc[nan_type_valid_area_cond, "MasVnrType"] = masvnrtype_mode

# apply same masonry transformations to test data using training values
true_missing_masonry_cond_test = x_test['MasVnrType'].isna() & (x_test['MasVnrArea'].isna() | (x_test['MasVnrArea'] == 0))
valid_type_nan_area_cond_test = ~(x_test['MasVnrType'].isna()) & (x_test['MasVnrArea'].isna() | (x_test['MasVnrArea'] == 0))
nan_type_valid_area_cond_test = x_test['MasVnrType'].isna() & ~(x_test['MasVnrArea'].isna() | (x_test['MasVnrArea'] == 0))

x_test.loc[true_missing_masonry_cond_test, "MasVnrType"] = "None"
x_test.loc[true_missing_masonry_cond_test, "MasVnrArea"] = 0

x_test.loc[valid_type_nan_area_cond_test, "MasVnrArea"] = masvnrarea_median
x_test.loc[nan_type_valid_area_cond_test, "MasVnrType"] = masvnrtype_mode

# features with missing values to be replaced with "None"
replace_with_none = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence"
]

# features with missing values to be replaced with 0
replace_with_0 = [
    "GarageYrBlt"
]

# features with missing values to be replaced with median
replace_with_median = [
    "LotFrontage",
]

# features to be dropped
features_to_drop = [
    "MiscFeature"
]

# imputing missing values
none_imputer = SimpleImputer(strategy="constant", fill_value="None")
x_train[replace_with_none] = none_imputer.fit_transform(x_train[replace_with_none])

zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
x_train[replace_with_0] = zero_imputer.fit_transform(x_train[replace_with_0])

median_imputer = SimpleImputer(strategy="median")
x_train[replace_with_median] = median_imputer.fit_transform(x_train[replace_with_median])

x_train.drop(columns=features_to_drop, inplace=True)

# transform test data with training-fitted imputers
x_test[replace_with_none] = none_imputer.transform(x_test[replace_with_none])
x_test[replace_with_0] = zero_imputer.transform(x_test[replace_with_0])
x_test[replace_with_median] = median_imputer.transform(x_test[replace_with_median])
x_test.drop(columns=features_to_drop, inplace=True)

""" -- Scaling and Encoding -- """

print("-- Features Sorting --\n")
numeric_features = x_train.select_dtypes(include="number").columns.tolist()
categorical_features = x_train.select_dtypes(include="object").columns.tolist()
print(f"Numeric Features\n{numeric_features}\n")
print(f"Categorical Features\n{categorical_features}\n")

scale_robust = []
scale_standard = []
for col in numeric_features:
    if x_train[col].skew() > 1:
        scale_robust.append(col)
    else:
        scale_standard.append(col)

print("-- Unique Values for Categorical Features --\n")
for col in categorical_features:
    print(f"{col}:{x_train[col].unique()}\n")

# features to be encoded using OHE
encode_ohe = [
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "Electrical",
    "Functional",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
    "Fence",
    "SaleType",
    "SaleCondition"
]

# features to be ordinaly encoded
encode_ordinal_mapping = {
    "ExterQual": ["Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Fa", "Gd", "Ex"]
}

# features to be binary encoded
encode_binary = [
    "CentralAir"
]

# build encoders
ohe_encoder = OneHotEncoder(sparse_output=False)
ordinal_encoder = OrdinalEncoder(categories=[encode_ordinal_mapping[col_name] for col_name in encode_ordinal_mapping])
binary_encoder = OneHotEncoder(drop="if_binary", sparse_output=False)

# build preprocessor to handle scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("standard_scl", StandardScaler(), scale_standard),
        ("robust_scl", RobustScaler(), scale_robust),
        ("ohe", ohe_encoder, encode_ohe),
        ("ordinal_enc", ordinal_encoder, [col_name for col_name in encode_ordinal_mapping]),
        ("binary_enc", binary_encoder, encode_binary)
    ]
)