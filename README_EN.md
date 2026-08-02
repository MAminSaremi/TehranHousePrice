# 🏠 Tehran House Price Prediction

Predicting house prices in Tehran using a CatBoost model, combined with domain-knowledge-driven feature engineering (municipal district mapping), leakage-free target encoding, and evaluation via cross-validation.

## Project Features

### Domain Knowledge Feature Engineering

- **`district_number`**: the raw dataset doesn't include the official municipal district number, even though this is one of the strongest price drivers in the real Tehran housing market. Each neighborhood was therefore manually mapped to its corresponding district.
- **`is_suburb`**: neighborhoods in Tehran's suburbs (e.g. Pardis, Islamshahr, Parand, etc.) were identified and flagged as a separate binary feature.
- **`top_area`**: EDA revealed that properties with an area over 120 square meters located in districts 1, 2, or 3 show a substantial price gap compared to the rest of the data. This nonlinear pattern was engineered as a dedicated feature so the model can explicitly learn this premium market segment.
- **`facility_score`**: a property's amenities (number of rooms, parking, elevator, warehouse) are summed into a single number, giving a simple numeric representation of its overall facility level.
- **`area_per_room`**: the ratio of total area to number of rooms, capturing average room size. Houses with the same total area but different room counts often have different market values, and this feature helps the model distinguish those cases.

### Preprocessing Decisions

- 23 missing values in the `Address` column are filled with `"missing"` (rather than dropping the record).
- 113 seemingly duplicate records are **intentionally kept**: in the real estate market, such records can represent different units or floors within the same building that naturally share the same address, area, and structural characteristics. Dropping them as duplicates would discard valid market information.
- `Address`, `district_number`, and `is_suburb` are passed to the model as **categorical** features (not numeric), since they are independent categories rather than ordered values. This lets the model learn a separate representation for each category instead of assuming any ordinal relationship between them.
- Before being treated as categorical, `district_number`'s missing values are filled with `0`, converted to integer (to remove decimal artifacts like `2.0` → `2`), then converted to string.

### Leakage-Free Target Encoding

- A smoothed target encoding is applied to `district_number`: each district's mean price is blended with the overall dataset mean, so districts with fewer samples get less weight in the final estimate (preventing overfitting on sparsely represented districts).
- This encoding is **always computed from the training data only** and then applied to the validation/test data — both in the main train/test split and independently within each cross-validation fold — to avoid data leakage.
- Districts not seen in the training data (unseen districts) are filled with the overall target mean (`global_mean`).

### Model Evaluation with 5-Fold Cross Validation

- The training data is split into 5 folds; in each iteration, one fold is used for validation and the remaining four for training.
- Target encoding is recomputed **independently for each fold, using only that fold's training portion**, so information from the validation fold never leaks into feature construction.
- The mean and standard deviation of the R² score across the five folds are reported as an estimate of the model's generalization performance and stability.

### Modeling

- Algorithm: **CatBoostRegressor** with `loss_function="RMSE"` and `early_stopping_rounds=100`.
- Key hyperparameters: `iterations=3000`, `learning_rate=0.02`, `depth=4`, `l2_leaf_reg=1`.
- The target variable (`Price`) is log-transformed with `log1p` to reduce the effect of its skewed distribution, and converted back with `expm1` during evaluation.

## Dataset

Required file: `tehranHprice.csv`

Main columns:

| Column                             | Description                         |
| ---------------------------------- | ----------------------------------- |
| `Area`                             | Square footage                      |
| `Room`                             | Number of rooms                     |
| `Parking`, `Warehouse`, `Elevator` | Property amenities                  |
| `Address`                          | Neighborhood                        |
| `Price`                            | Price in Toman — target variable    |
| `Price(USD)`                       | Price in USD (not used in training) |

Engineered columns: `district_number`, `is_suburb`, `top_area`, `facility_score`, `area_per_room`, `district_target_mean`

## Installation

```bash
pip install pandas numpy scikit-learn catboost matplotlib
```

## How to Run

Place `tehranHprice.csv` in the same directory as the script, then run:

```bash
python main.py
```

The output includes the R² score for each fold, their mean and standard deviation, the final RMSE and R² on the test set, and a feature importance table.

## Model Results

Results logged in the code (comment at the end of `main.py`):

| Metric      | Value         |
| ----------- | ------------- |
| Mean CV R²  | 0.8829        |
| Std CV R²   | 0.0235        |
| RMSE (Test) | 2,453,876,315 |
| R² (Train)  | 0.9038        |
| R² (Test)   | 0.8573        |

### Feature Importance

| Feature                | Importance |
| ---------------------- | ---------- |
| `Area`                 | 24.73      |
| `Address`              | 18.00      |
| `district_number`      | 15.79      |
| `district_target_mean` | 12.06      |
| `is_suburb`            | 9.16       |
| `facility_score`       | 6.54       |
| `area_per_room`        | 5.10       |
| `top_area`             | 4.15       |
| `Room`                 | 2.62       |
| `Parking`              | 0.86       |
| `Elevator`             | 0.62       |
| `Warehouse`            | 0.36       |

Area is the single most important factor, but location-related features (`Address`, `district_number`, `district_target_mean`, `is_suburb`) together account for a very large share of predictive importance right after it — consistent with the well-known role of location in the Tehran housing market.

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn (KFold, train_test_split, r2_score, mean_squared_error)
- CatBoost
- Matplotlib

## Project Structure

```
.
├── main.py            # Main script: feature engineering, target encoding, model training and evaluation
├── tehranHprice.csv    # Dataset (must be provided separately)
└── README.md
```

## Additional Notes

- The neighborhood-to-district mapping (`neighborhood_to_district`) and the list of suburban neighborhoods (`suburb`) were built manually, based on real knowledge of Tehran's municipal district boundaries, not derived from the dataset itself.
- The `target_encoding` function is called separately once for the main train/test split and once inside each cross-validation fold; this separation ensures that the statistics computed never use any validation/test data they're applied to.
- Keeping the duplicate records and filling missing `Address` values with `"missing"` (instead of dropping them) were both deliberate choices made to avoid losing real market information.

## License

This project was created for educational/personal purposes.
