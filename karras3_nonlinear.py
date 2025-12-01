# CS 482 - Assignment 3
# Author: Demetri Karras
# File: karras3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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

X = X.drop(columns=["Id"], axis=1)

# obtain features that have missing values along with all their unique values
print("-- Features with NaN Values --\n")
missing_value_features = X.columns[X.isna().any()]
for mvf in missing_value_features:

    total_missing = X[mvf].isna().sum()
    percent_missing = round(X[mvf].isna().mean() * 100, 2) # quantity of NaN values for a feature multiplied by 100 and rounded to two decimals

    print(f"{mvf}: {total_missing} ({percent_missing}%)")
    # print(f"Unique Values: {X[mvf].unique()}\n")
print()

# metrics for masonry veneers - need to take a closer look at related features after seeing missing value counts
print("-- Metrics for Masonry Veneers --")
print(f"N/A Type Total: {X['MasVnrType'].isna().sum()}")
print(f"N/A Area Total: {((X['MasVnrArea'].isna()) | (X['MasVnrArea'] == 0)).sum()}\n")

true_missing_masonry_cond = X['MasVnrType'].isna() & (X['MasVnrArea'].isna() | (X['MasVnrArea'] == 0))
valid_type_nan_area_cond = ~(X['MasVnrType'].isna()) & (X['MasVnrArea'].isna() | (X['MasVnrArea'] == 0))
nan_type_valid_area_cond = X['MasVnrType'].isna() & ~(X['MasVnrArea'].isna() | (X['MasVnrArea'] == 0))
print(f"N/A Type + N/A Area Count: {true_missing_masonry_cond.sum()}") # truly missing masonry veneers
print(f"Valid Type + N/A Area Count: {valid_type_nan_area_cond.sum()}") # area not calculated
print(f"N/A Type + Valid Area Count: {nan_type_valid_area_cond.sum()}") # type not recorded


X.loc[true_missing_masonry_cond, "MasVnrType"] = "None"
X.loc[true_missing_masonry_cond, "MasVnrArea"] = 0
masvnrtype_mode = X.loc[X["MasVnrType"].notna(), "MasVnrType"].mode()[0]
X.loc[nan_type_valid_area_cond, "MasVnrType"] = masvnrtype_mode
masvnrarea_median = X.loc[(X["MasVnrArea"].notna()) & (X["MasVnrArea"] > 0), "MasVnrArea"].median()
X.loc[valid_type_nan_area_cond, "MasVnrArea"] = masvnrarea_median

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
X[replace_with_none] = none_imputer.fit_transform(X[replace_with_none])

zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
X[replace_with_0] = zero_imputer.fit_transform(X[replace_with_0])

median_imputer = SimpleImputer(strategy="median")
X[replace_with_median] = median_imputer.fit_transform(X[replace_with_median])

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()
