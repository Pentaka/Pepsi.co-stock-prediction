# PepsiCo Stock Prediction
> **This Project is not financial advice!!!**

In this project, a prediction model is created using the Random Forest algorithm based on Pepsi stock data. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/krupalpatel07/pepsico-daily-data).

## Exploratory Data Analysis (EDA)
The dataset contains the following columns: 'Open', 'High', 'Low', 'Close', and 'Volume'. Additionally, the features MA100 (100-day moving average), MA200 (200-day moving average), Price_Change, and Volume_Change have been added.

The chart below shows the changes in Pepsi's stock close prices over time:
![Close Price Over Time](https://github.com/Pentaka/Pepsi.co-stock-prediction/blob/main/Close%20Price%20Over%20Time.png)

It can be observed that over 50 years, the close prices steadily rose, with small fluctuations, surpassing 175 units.

Examining the daily percentage change in close prices:
![Daily Percentage Return](https://github.com/Pentaka/Pepsi.co-stock-prediction/blob/main/Daily%20Percentage%20Return.png)

The chart shows significant movement between 1980-1990. At the beginning of 2020, a notable increase in price volatility can be seen, likely due to the pandemic. Post-2020, there is continued sporadic movement.

## Correlation Matrix
![Correlation Matrix](https://github.com/Pentaka/Pepsi.co-stock-prediction/blob/main/Correlation%20Matrix.png)

From the correlation matrix, we observe that the features are ordered in terms of their correlation with the 'Close' value, from most positive to most negative:
- Close
- Open
- High
- Low
- MA100
- MA200
- Year
- Price_Change
- Day
- Month
- Volume_Change

## Training the Model
Before training, the data is normalized to be suitable for model training. RandomizedSearchCV is used to determine the optimal parameters for the Random Forest model. The best parameters found are:

Best Hyperparameters: {'n_estimators': 750, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 25}


After training with these parameters, the following results were obtained:
- **Mean Squared Error (MSE):** 303813879289.4307
- **Mean Absolute Error (MAE):** 159895.8676302513
- **R² Score:** 0.9562179430671003

## Prediction Evaluation
![Actual vs Predicted](https://github.com/Pentaka/Pepsi.co-stock-prediction/blob/main/Actual%20vs%20Predicted.png)

When comparing actual and predicted values, we see that the model generates reasonable predictions. However, since stock market movements are influenced by unpredictable external factors, it is not possible to achieve 100% accuracy.

## Libraries Used
The following libraries are required to run the project:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
