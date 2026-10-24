import pandas as pd
import numpy as np
# for data preprocessing
from sklearn.model_selection import train_test_split
# for converting text into numerical representation
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop the unnamed index column if it exists
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# Drop CustomerID as it's a unique identifier (not useful for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handle missing values
print("\nHandling missing values...")
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
df.Gender = df.Gender.replace("Fe male","Female")
df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})
# Merge Unmarried with Single, keep Single as category values of MaritalStatus
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried","Single")

# Encode categorical columns
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()

# List of categorical columns to encode
# categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
#                         'MaritalStatus', 'Designation']

# for col in categorical_features:
#     if col in df.columns:
#         df[col] = label_encoder.fit_transform(df[col].astype(str))

# Split into feature and target variables
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Type values kept as:", sorted(y.unique()))
print("Data prepared: train/test splits written.")
