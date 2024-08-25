import pandas as pd
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split

from src.data.load_data import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocess import Preprocessor
from src.models.selection import ModelSelection
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


def main():
    # Define dataset path
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    # Define model path
    model_path = 'models/house_price_regressor'

    # Define submission path
    submission_path = 'submission.csv'

    # Load data
    train_loader = DataLoader(train_path)
    test_loader = DataLoader(test_path)
    train_df = train_loader.load_data()
    test_df = test_loader.load_data()

    # Feature engineer data
    train_fe = FeatureEngineer(train_df)
    test_fe = FeatureEngineer(test_df)
    train_df = train_fe.engineer_features()
    test_df = test_fe.engineer_features()

    # Preprocess data
    train_proc = Preprocessor(train_df)
    test_proc = Preprocessor(test_df)
    train_df = train_proc.preprocess_data()
    test_df = test_proc.preprocess_data()

    # Remove ID columns
    train_df.drop(['Id'], axis=1, inplace=True)

    # Extract test data IDs
    ids = test_df['Id']
    test_df.drop(['Id'], axis=1, inplace=True)

    # Split training data
    X_train, X_eval, y_train, y_eval = train_test_split(
        train_df.drop(['SalePrice'], axis=1, inplace=False), train_df['SalePrice'], test_size=0.2, random_state=42)

    # Select best model
    selector = ModelSelection(X_train, y_train)

    # Define parameter grid for XGBRegressor
    xgb_param_grid = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.005, 0.01, 0.1],
        'max_depth': [6, 8, 12],
        'min_child_weight': [8, 9, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_lambda': [0, 0.5],
        'reg_alpha': [0, 0.2]
    }

    # Create an XGBRegressor instance
    xgb_model = XGBRegressor(n_jobs=-1)

    # Tune XGBRegressor
    xgb_model = selector.tune_model(xgb_model, xgb_param_grid)

    # Train model
    trainer = ModelTrainer(xgb_model)
    trainer.train(X_train, y_train)
    trainer.save_model(model_path)

    # Evaluate model
    evaluator = ModelEvaluator(trainer.model)
    evaluator.evaluate(X_eval, y_eval)

    # Assuming X_train and X_test are your training and test dataframes
    train_columns = set(X_train.columns)
    test_columns = set(test_df.columns)

    # Columns present in training but missing in test
    missing_in_test = train_columns - test_columns

    # Columns present in test but missing in training
    missing_in_train = test_columns - train_columns

    # Remove columns from test data that were not in training data
    test_df = test_df.drop(columns=missing_in_train)

    # Add missing columns to test data with all values set to 0
    for col in missing_in_test:
        test_df[col] = 0

    # Ensure the columns in the same order
    test_df = test_df[X_train.columns]

    # Get predictions on test dataset
    predictions = trainer.model.predict(test_df)

    # Create data frame for submission
    submission_df = pd.DataFrame({
        'Id': ids,
        'SalePrice': predictions
    })

    # Output submission.csv file
    submission_df.to_csv(submission_path, index=False)


if __name__ == '__main__':
    main()
