from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)

        # Print metrics
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        print(f"Median Absolute Error: {median_ae:.4f}")
