from xgboost.sklearn import XGBRegressor


class ModelTrainer:
    def __init__(self, model=None):
        self.model = model

    def train(self, X_train, y_train):
        if self.model is None:
            self.model = XGBRegressor(
                n_estimators=1500,  # Number of boosting rounds
                learning_rate=0.01,  # Step size for each boosting round
                max_depth=8,  # Maximum depth of each tree
                min_child_weight=10,  # Minimum sum of instance weight needed in a child
                subsample=0.8,  # Fraction of samples to use for each tree
                colsample_bytree=0.8,  # Fraction of features to use for each tree
                reg_lambda=0.5,  # L2 regularization term
                reg_alpha=0.2,  # L1 regularization term
                n_jobs=-1  # Use all available cores
            )
        self.model.fit(X_train, y_train)

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
