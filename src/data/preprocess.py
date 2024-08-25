import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Initialize encoders for each categorical feature
        self.encoders = {
            'MSSubClass': OneHotEncoder(),
            'MSZoning': OneHotEncoder(),
            'Street': LabelEncoder(),
            'LotShape': OneHotEncoder(),
            'LandContour': LabelEncoder(),
            'Utilities': LabelEncoder(),
            'LotConfig': OneHotEncoder(),
            'LandSlope': LabelEncoder(),
            'Neighborhood': OneHotEncoder(),
            'Condition1': OneHotEncoder(),
            'BldgType': OneHotEncoder(),
            'HouseStyle': OneHotEncoder(),
            'RoofStyle': OneHotEncoder(),
            'RoofMatl': OneHotEncoder(),
            'Exterior1st': OneHotEncoder(),
            'Exterior2nd': OneHotEncoder(),
            'OverallQual': LabelEncoder(),
            'OverallCond': LabelEncoder(),
            'MasVnrType': OneHotEncoder(),
            'ExterQual': LabelEncoder(),
            'ExterCond': LabelEncoder(),
            'Foundation': OneHotEncoder(),
            'BsmtQual': LabelEncoder(),
            'BsmtCond': LabelEncoder(),
            'BsmtExposure': LabelEncoder(),
            'BsmtFinType1': OneHotEncoder(),
            'BsmtFinType2': OneHotEncoder(),
            'Heating': OneHotEncoder(),
            'HeatingQC': LabelEncoder(),
            'CentralAir': LabelEncoder(),
            'Electrical': OneHotEncoder(),
            'KitchenQual': LabelEncoder(),
            'Functional': LabelEncoder(),
            'FireplaceQu': LabelEncoder(),
            'GarageType': OneHotEncoder(),
            'GarageFinish': OneHotEncoder(),
            'GarageQual': LabelEncoder(),
            'GarageCond': LabelEncoder(),
            'PavedDrive': LabelEncoder(),
            'PoolQC': LabelEncoder(),
            'Fence': LabelEncoder(),
            'MiscFeature': OneHotEncoder(),
            'MoSold': OneHotEncoder(),
            'SaleType': OneHotEncoder(),
            'SaleCondition': OneHotEncoder()
        }

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                if isinstance(encoder, LabelEncoder):
                    df[feature] = encoder.fit_transform(df[feature])
                elif isinstance(encoder, OneHotEncoder):
                    encoded = encoder.fit_transform(df[[feature]]).toarray()
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([feature]))
                    df = pd.concat([df, encoded_df], axis=1).drop(feature, axis=1)
        return df

    def preprocess_data(self) -> pd.DataFrame:
        self.df = self.encode_features(self.df)
        return self.df
