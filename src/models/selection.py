from sklearn.model_selection import GridSearchCV


class ModelSelection:
    def __init__(self, X_train, y_train, cv=3):
        """
        Initialize the ModelSelection with training data and cross-validation setting.

        Parameters:
        - X_train: Training features dataframe
        - y_train: Training target variable
        - cv: Number of cross-validation folds
        """
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune_model(self, model, param_grid, scoring='neg_mean_squared_error'):
        """
        Tune the model using Grid Search with Cross-Validation.

        Parameters:
        - model: The machine learning model to be tuned
        - param_grid: Dictionary with parameters to try
        - scoring: Scoring metric for model evaluation

        Returns:
        - The best model found
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=self.cv,
            n_jobs=-1  # Use all available cores
        )
        grid_search.fit(self.X_train, self.y_train)

        # Store best parameters and model
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = -grid_search.best_score_  # Convert back to positive score

        print("Best parameters found: ", self.best_params)
        print("Best score found: ", self.best_score)

        return self.best_model
