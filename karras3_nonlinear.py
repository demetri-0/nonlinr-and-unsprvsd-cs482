# CS 482 - Assignment 3
# Author: Demetri Karras
# File: karras3_nonlinear.py

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR

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
x_train, x_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    shuffle=True, 
    random_state=42
)

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
    skew_val = x_train[col].skew()
    nonzero_ratio = (x_train[col] != 0).mean()   # percentage of non-zero values

    if skew_val > 1 and nonzero_ratio > 0.2:
        scale_robust.append(col)
    else:
        scale_standard.append(col)

print(scale_robust)
print(scale_standard)

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
ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal_encoder = OrdinalEncoder(categories=[encode_ordinal_mapping[col_name] for col_name in encode_ordinal_mapping])
binary_encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)

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

pca = PCA(n_components=10)

# fit and transform processor on training, transform test
x_train_transformed = preprocessor.fit_transform(x_train)
x_test_transformed = preprocessor.transform(x_test)

# fit and transform PCA on training, transform test
x_train_pca = pca.fit_transform(x_train_transformed)
x_test_pca = pca.transform(x_test_transformed)

# reform training data to DataFrame
x_train = pd.DataFrame(
    x_train_pca,
    columns=pca.get_feature_names_out()
)

# reform training data to DataFrame
x_test = pd.DataFrame(
    x_test_pca,
    columns=pca.get_feature_names_out()
)

print(" -- PCA Contributing Features -- \n")
# build DataFrame table and display PCA contributing features
preproc_feature_names = preprocessor.get_feature_names_out()
loading_df = pd.DataFrame(
    pca.components_[:2],
    columns=preproc_feature_names,
    index=["PC1", "PC2"]
)
print(round(loading_df.loc["PC1"].abs(), 2).sort_values(ascending=False).head(10))
print()
print(round(loading_df.loc["PC2"].abs(), 2).sort_values(ascending=False).head(10))
print()

# split training data into sub-training set and validation set
x_train_sub, x_val, y_train_sub, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.20, 
    shuffle=True, 
    random_state=42
)

# test different parameters for SVM and print results
best_rmse = float("inf")
best_params = None
for gamma in range(1, 11, 2):
    for cost in range(10, 101, 10):

        model = SVR(kernel="rbf", gamma=gamma, C=cost)
        model.fit(x_train_sub, y_train_sub)
        
        y_pred = model.predict(x_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_params = (gamma, cost)

print(" -- SVM Tuning Results -- \n")
print(f"Best Gamma: {best_params[0]}\nBest Cost: {best_params[1]}")
print(f"Best RMSE: {round(best_rmse, 2)}\nBest R^2: {round(best_r2, 2)}\n")

# test different parameters for NN and print results
best_rmse = float("inf")
best_num_units = 0
for num_units in range(10, 31, 2):

    model = MLPRegressor(hidden_layer_sizes=(num_units,), random_state=42)
    model.fit(x_train_sub, y_train_sub)

    y_pred = model.predict(x_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_num_units = num_units

print(" -- NN Tuning Results -- \n")
print(f"Best # of Hidden Units: {best_num_units}")
print(f"Best RMSE: {round(best_rmse, 2)}\nBest R^2: {round(best_r2, 2)}\n")

""" ******** Model Evaluation ******** """

""" -- SVM Evaluation -- """

best_gamma, best_cost = best_params

# create final SVM model for evaluation, fitting on training data
svm = SVR(kernel="rbf", gamma=best_gamma, C=best_cost)
svm.fit(x_train, y_train)

svm_y_test_pred = svm.predict(x_test)

# get and display evaluation metrics for SVM
svm_test_rmse = root_mean_squared_error(y_test, svm_y_test_pred)
svm_test_r2 = r2_score(y_test, svm_y_test_pred)

print(" -- SVM Evaluation -- \n")
print("SVM Test RMSE:", round(svm_test_rmse, 2))
print("SVM Test R^2:", round(svm_test_r2, 2))

""" -- NN Evaluation -- """

# create final NN model for evaluation, fitting on training data
nn = MLPRegressor(
    hidden_layer_sizes=(best_num_units,),
    random_state=42
)
nn.fit(x_train, y_train)

nn_y_test_pred = nn.predict(x_test)

# get and display evaluation metrics for NN
nn_test_rmse = root_mean_squared_error(y_test, nn_y_test_pred)
nn_test_r2 = r2_score(y_test, nn_y_test_pred)

print(" -- NN Evaluation -- \n")
print("NN Test RMSE:", round(nn_test_rmse, 2))
print("NN Test R^2:", round(nn_test_r2, 2))
