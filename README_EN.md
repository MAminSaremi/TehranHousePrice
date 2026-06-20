# 🏠 Tehran House Price Prediction

A machine learning project that predicts house prices in Tehran using CatBoost, XGBoost, and Random Forest, based on features such as area, number of rooms, parking, warehouse, elevator, and neighborhood (Address).

## 📋 Features

- Data cleaning: removal of missing values and duplicate records
- Removal of invalid prices (Price ≤ 0)
- Log transformation of price (`log1p`) to improve model accuracy, since the price distribution is skewed and the data can't simply be removed as outliers without losing important rows
- Handling categorical values (`Address`) with `pd.get_dummies` inside the hyperparameter search function (`GridSearchCV`)
- Hyperparameter tuning with `GridSearchCV` for three models:
  - XGBoost
  - CatBoost ✅ (final selected model)
  - Random Forest
- Model evaluation using `R²` and `RMSE`
- Interactive CLI section for predicting the price of a new house from user input
- Conversion of the predicted price to USD based on a user-provided exchange rate

## 📊 Dataset

Required file: `tehranHprice.csv`

Main columns used:

| Column | Description |
| --- | --- |
| `Area` | Square footage |
| `Room` | Number of rooms |
| `Parking` | Has parking (0/1) |
| `Warehouse` | Has warehouse (0/1) |
| `Elevator` | Has elevator (0/1) |
| `Address` | Neighborhood/district |
| `Price` | Price in Toman — target variable |
| `Price(USD)` | Price in USD (not used in training, since it has no meaningful impact on the model) |

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn catboost xgboost
```

## 🚀 How to Run

1. Place the `tehranHprice.csv` file in the same directory as the script.
2. Run the script:

```bash
python main.py
```

3. The CatBoost model is trained automatically.
4. The program then interactively asks for the details of a new house:

```
Example:
area : 120
room : 2
parking(0/1) : 1
Warehouse(0/1) : 1
Elevator(0/1) : 1
Address : Niavaran
Current USD rate: 60000
```

5. The output includes the predicted price in Toman and its equivalent in USD.

> ⚠️ Note: The `Address` value must exactly match one of the neighborhoods present in the dataset; otherwise an error message is shown.

## 📈 Model Results (GridSearchCV + pd.get_dummies)

| Model | R² (Train) | R² (Test) | RMSE (Test) |
| --- | --- | --- | --- |
| XGBoost | 84.56% | 67.94% | 318,478,591 |
| CatBoost | 86.23% | 68.71% | 314,591,703 |
| Random Forest | 93.52% | 69.30% | 311,630,290 |

The final model used in this project is **CatBoost** with the following tuned parameters:

```
iterations=500, depth=4, l2_leaf_reg=1, learning_rate=0.1, colsample_bylevel=1, subsample=1
```

> ⚠️ Note: Since CatBoost can natively handle categorical features, the `pd.get_dummies` step is commented out when training the final model, and the `cat_features` parameter is used instead.

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- CatBoost
- XGBoost

## 📁 Project Structure

```
.
├── main.py              # Main project script
├── tehranHprice.csv     # Dataset (must be provided separately)
└── README.md
```

## 📝 Additional Notes

- The `GridSearchCV` blocks for all three models are kept commented out in the code so they can be re-run if needed (they are time-consuming).
- The target variable (`Price`) is log-transformed with `log1p` and converted back to the original scale with `expm1` during evaluation.
- The `R²` and `RMSE` metrics measure the model's accuracy in predicting the price value, not its accuracy in fitting the model's internal algorithmic equations.

## 📄 License

This project was created for educational/personal purposes.
