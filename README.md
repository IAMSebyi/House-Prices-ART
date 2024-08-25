# House Price Prediction

This project focuses on predicting house prices using machine learning techniques for the ["House Prices - Advanced Regression Techniques" Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The goal was to develop a robust regression model to estimate house prices based on various features provided in the dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains a complete pipeline for predicting house prices. The process includes data loading, feature engineering, preprocessing, model selection, training, and evaluation. The main model used in this project is an XGBoost Regressor, which was fine-tuned to achieve optimal performance.

### Achievements

- **Leaderboard Ranking:** Achieved a position in the Top 24% of participants on the Kaggle leaderboard.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/your_username/house-price-prediction.git
    cd house-price-prediction
    ```

2. Create and activate a Python virtual environment (optional but recommended):
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized into the following directories and files:

```
.
├── data/
│   ├── train.csv
│   ├── test.csv
├── models/
│   ├── house_price_regressor
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── feature_engineer.py
│   │   ├── preprocess.py
│   ├── models/
│   │   ├── selection.py
│   │   ├── train.py
│   │   ├── evaluate.py
├── submission.csv
├── main.py
├── README.md
├── requirements.txt

```


- **data/**: Contains the training and testing datasets.
- **models/**: Stores the trained model.
- **src/data/**: Contains scripts for data loading, feature engineering, and preprocessing.
- **src/models/**: Contains scripts for model selection, training, and evaluation.
- **submission.csv**: The final submission file with predictions.
- **main.py**: The main script to run the pipeline.

## Usage

To train the model and make predictions, run the `main.py` script:

```
python main.py
```


This will load the data, perform feature engineering, preprocess the data, train the model, and save the predictions to `submission.csv`.

## Results

The predictions on the test data are saved in `submission.csv` in the following format:

```
Id,SalePrice
1,208500.0
2,181500.0
3,223500.0
...

```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request to contribute to this project.

## License

This project is licensed under the MIT License.
