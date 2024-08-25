import pandas as pd


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def drop_irrelevant_features(df: pd.DataFrame) -> pd.DataFrame:
        irrelevant_features = [
                               # Highly Redundant or Less Informative Features
                               'Condition2', 'MasVnrArea', 'Fireplaces',
                               'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                               # Features with Significant Overlap with Other Features
                               'GarageYrBlt', 'PoolArea',
                               'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                               # Features Less Likely to Impact SalePrice
                               'Alley', 'MiscVal'
        ]
        df.drop(irrelevant_features, axis=1, inplace=True)
        return df

    def engineer_features(self) -> pd.DataFrame:
        self.df = self.drop_irrelevant_features(self.df)
        return self.df
