import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("tehranHprice.csv")

# print(df['Address'].unique())

#? The original dataset does not include the official district number for each address.
#? However, in the real Tehran housing market, the district is one of the strongest factors
#? affecting property prices. Therefore, I manually mapped each address to its corresponding
#? district and added it as a new feature. Additionally, addresses located in suburban areas
#? around Tehran were identified and added as a separate binary feature.

neighborhood_to_district = {
    # منطقه ۱
    'Velenjak': 1, 'Zaferanieh': 1, 'Niavaran': 1, 'Kamranieh': 1, 'Aqdasieh': 1,'Sohanak':1,
    'Darband': 1, 'Darakeh': 1, 'Tajrish': 1, 'Farmanieh': 1, 'Ozgol': 1,'Gheitarieh':1,
    'Araj': 1, 'Dezashib': 1, 'Ekhtiarieh': 1, 'Hekmat': 1,'Mahallati':1,'Elahieh':1,
    'Mahmoudieh':1,'Ajudaniye':1,'Chidz':1,

    # منطقه ۲
    'Shahrake Gharb': 2, 'Marzdaran': 2, 'Punak': 2,'Sadeghieh': 2,'Shahrake Qods':2,'Saadat Abad':2,'ShahrAra':2,'Gisha':2,
    'Sattarkhan': 2,'Daryan No':2,'Tarasht':2,'Telecommunication':2,'Shahrake Quds':2,

    # منطقه ۳
    'Vanak': 3, 'Jordan': 3, 'Mirdamad': 3, 
    'Zafar': 3, 'Gholhak': 3, 'Zargandeh': 3,'Dorous': 3,
    'Pasdaran':3,'Mirdamad':3,'Jordan':3,
    'Vanak':3,'Ghoba':3,'Seyed Khandan':3,

    # منطقه ۴
    'Lavizan': 4, 'Hakimiyeh': 4,'West Pars':4,'Shams Abad':4,'Elm-o-Sanat':4,
    'Heravi':4,'East Pars':4,'Mehran':4,'Kazemabad':4,

    # منطقه ۵
    'Ekbatan': 5,'Shahran':5,'Andisheh':5,'Feiz Garden':5,'Water Organization':5,'Chardivari':5,'Qalandari':5,'Abazar':5,'Shahrakeh Naft':5,
    'Koohsar':5,'Shahrake Apadana': 5,'Shahr-e-Ziba': 5,'West Ferdows Boulevard': 5, 'East Ferdows Boulevard': 5,'Central Janatabad': 5, 'Northern Janatabad': 5, 'Southern Janatabad': 5,
    'Eram':5,'North Program Organization':5,'Southern Program Organization':5,

    # منطقه ۶
    'Yousef Abad': 6, 'Amirabad': 6, 'Keshavarz Boulevard': 6, 'Fatemi': 6,'Mirza Shirazi':6,'Argentina':6,
    'Karimkhan': 6,'Valiasr':6,'Amirabad':6,'Villa':6,'Yousef Abad':6,'Gandhi':6,

    # منطقه ۷
    'Abbasabad': 7, 'Nezamabad': 7, 'Heshmatieh': 7,'Bahar':7,'Haft Tir':7,'Northern Suhrawardi':7,'Garden of Saba':7,
    
    # 9
    'Ostad Moein' : 9,'Si Metri Ji':9,
    # منطقه ۸
    'Narmak': 8, 'Majidieh': 8, 'Tehran Now': 8, 'Sabalan': 8,'Vahidieh':8,'Shahrake Madaen':8,'Taslihat':8,
    'Vahidiyeh':8,

    # منطقه ۱۰
    'Beryanak': 10,'Salsabil':10,'Qasr-od-Dasht':10,'Jeyhoon':10,'Hashemi':10,'Karoon':10,'Komeil':10,
    'Nawab':10,

    # منطقه ۱۱
    'Enghelab': 11, 'Republic': 11, 'Amirieh': 11,'Moniriyeh' :11,'Northren Jamalzadeh':11,'Azarbaijan':11,'Amir Bahador':11,
    'Razi':11,'Eskandari':11,

    # منطقه ۱۲
    'Waterfall':12,'Hassan Abad':12,

    # منطقه ۱۳
    'Pirouzi': 13,'Air force':13,'Thirteen November':13,
    # 14
    'Parastar':14,'Ahang':14,

    # منطقه ۱۵
    'Afsarieh': 15, 'Khavaran': 15,'Atabak':15,

    # منطقه ۱۶
    'Naziabad': 16,'Railway':16,'Shoosh':16,'Aliabad South':16,'Javadiyeh':16,'Yakhchiabad':16,
    # 17
    'Qazvin Imamzadeh Hassan':17,'Fallah':17,'Azari':17,'Boloorsazi':17,
    # 18
    'Yaftabad':18,'Shadabad':18,
    # 19
    'Baghestan':19,'Salehabad':19,
    # منطقه ۲۰
    'Ray': 20, 'Ray - Montazeri': 20, 'Ray - Pilgosh': 20,
    # 21
    'Tehransar':21,'Shahrake Azadi':21,
    # منطقه ۲۲
    'Northern Chitgar': 22, 'Southern Chitgar': 22, 'Dehkade Olampic': 22,'Shahrake Shahid Bagheri':22,'Persian Gulf Martyrs Lake':22
    ,'Golestan':22,'Southern Chitgar':22,'Kook':22,'Azadshahr':22,'Zibadasht':22
}
df['district_number'] = df['Address'].map(neighborhood_to_district)  

suburb={
    'Pardis',
    'Islamshahr',
    'Parand',
    'Pakdasht',
    'Pakdasht KhatunAbad',
    'Shahryar',
    'Rudhen',
    'Chahardangeh',
    'Baqershahr',
    'Kahrizak',
    'Qarchak',
    'Damavand',
    'Absard',
    'Lavasan',
    'Shahedshahr',
    'Nasim Shahr',
    'Chardangeh',
    'Tenant',
    'Malard',
    'SabaShahr',
    'Pishva',
    'Islamshahr Elahieh',
    'Ray - Montazeri',
    'Firoozkooh Kuhsar',
    'Robat Karim',
    'Ray - Pilgosh',
    'Ghiyamdasht',
    'Safadasht',
    'Khademabad Garden',
    'Mehrabad River River',
    'Varamin - Beheshti',
    'Alborz Complex',
    'Firoozkooh'
}
pattern = '|'.join(suburb)
df['is_suburb'] = df['Address'].str.contains(pattern,case=False, na=False)

#? EDA revealed a premium housing segment characterized by:
#? - Area > 120 square meters
#? - Located in districts 1, 2, or 3
#?
#? Properties satisfying these conditions showed a substantial price gap compared to
#? the remaining observations. To help the model explicitly learn this nonlinear
#? market behavior, a dedicated feature was engineered to identify this premium group.
df["top_area"] = False
mask = (
    (df["Area"] > 120) &
    (df["district_number"].isin([1, 2, 3]))
)
df.loc[mask, "top_area"] = True

#? To help the model better capture the effect of property amenities on price,
#? a facility_score feature was created by summing the availability of key
#? amenities (Room,parking, elevator, and warehouse). This provides a simple
#? numerical representation of the overall facility level of each property.
df['facility_score'] = 0
facility_score = (
    df["Room"] +
    df["Parking"].astype(int) +
    df["Elevator"].astype(int) +
    df["Warehouse"].astype(int)
)
df['facility_score'] = facility_score

#? A new feature (area_per_room) was engineered to capture the average room size.
#? While the dataset already contains both total area and number of rooms, this ratio
#? provides additional information about the property's layout. Houses with the same
#? total area but different room counts often have different market values, and this
#? feature helps the model distinguish those cases.
df['area_per_room']= df['Area'] / df['Room']

print(df.head(20))

# plt.figure(figsize=(12, 5))
# plt.bar(district_stats.index.astype(str), district_stats['mean'])
# plt.xlabel('شماره منطقه')
# plt.ylabel('میانگین قیمت')
# plt.title('میانگین قیمت به تفکیک منطقه')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()


#? WE HAVE 23 MISSINGVALUES IN ADDRESS COLUMN
# print(df.loc[df['Address'].isna()])
df['Address'] = df['Address'].fillna('missing')

#? we have 113 duplicated data
#? Although the dataset contains a number of seemingly duplicate records, they were
#? intentionally retained. In the real estate market, such records can represent
#? different units or floors within the same building, where properties naturally
#? share the same address, floor area, and structural characteristics. Removing
#? these observations as duplicates would discard valid information and could
#? negatively affect the model's ability to learn real market patterns.
# df = df.drop_duplicates()


df = df[df["Price"] > 0]

#? District number and suburb status are passed as categorical features since
#? they are independent categories, not ordered numerical values. This allows
#? the model to learn a separate representation for each category instead of
#? assuming any ordinal relationship between them.
cat_feature = ["Address","district_number", "is_suburb"]

#? District numbers are categorical identifiers rather than numerical values.
#? Missing values are replaced with "0", then converted to integers to remove
#? decimal representations (e.g., 2.0 -> 2), and finally converted to strings
#? so they are treated as categorical values instead of continuous numerical features.
df["district_number"] = (
    df["district_number"]
      .fillna(0)
      .astype(int)
      .astype(str)
)

x = df.drop(columns=["Price" , "Price(USD)"])


# !print((df["Price"] / df["Price(USD)"]).describe())
# !
# !print(df[["Price", "Price(USD)"]].corr())

y = np.log1p(df["Price"])

model_reg = CatBoostRegressor(  iterations=3000,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=1,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=False)


train_x , test_x , train_y , test_y = train_test_split(x , y , test_size=0.1 , random_state=42)


def target_encoding(X_train, X_valid, y_train, m=50):

    # Create a temporary dataframe containing features and target
    train_df = X_train.copy()
    train_df["target"] = y_train

    # Overall target mean (used for smoothing and unseen districts)
    global_mean = train_df["target"].mean()

    # Compute district-wise target statistics
    stats = train_df.groupby("district_number")["target"].agg(["mean","count"])

    # Apply smoothing to avoid overfitting on districts with few samples
    stats["smooth"] = (
        stats["count"]*stats["mean"] + m*global_mean
    )/(stats["count"]+m)

     # Copy datasets before adding the new feature
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    # Map the smoothed target mean to each district
    X_train["district_target_mean"] = (X_train["district_number"].map(stats["smooth"]))
    X_valid["district_target_mean"] = (X_valid["district_number"].map(stats["smooth"]))

    # Replace unseen districts with the global target mean
    X_train["district_target_mean"] = X_train["district_target_mean"].fillna(global_mean)
    X_valid["district_target_mean"] = X_valid["district_target_mean"].fillna(global_mean)

    return X_train, X_valid

train_x, test_x = target_encoding(
    train_x,
    test_x,
    train_y
)

#? 5-Fold Cross Validation
#?
#? The training data is split into 5 different folds.
#? In each iteration:
#?   1. One fold is used as the validation set.
#?   2. The remaining four folds are used for training.
#?   3. Target Encoding is computed ONLY from the training fold to prevent
#?      data leakage.
#?   4. CatBoost is trained on the training fold.
#?   5. The model is evaluated on the validation fold using the R² metric.
#?
#? Finally, the mean and standard deviation of the five R² scores are used
#? to estimate the model's generalization performance and stability.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store the R² score of each validation fold
scores = []

for train_idx, valid_idx in kf.split(train_x):

    # Split the current fold into training and validation subsets
    X_train = train_x.iloc[train_idx].copy()
    X_valid = train_x.iloc[valid_idx].copy()

    y_train = train_y.iloc[train_idx]
    y_valid = train_y.iloc[valid_idx]

    # Create leakage-free target encoding using only the current training fold
    X_train, X_valid = target_encoding(
        X_train,
        X_valid,
        y_train
    )

    # Train CatBoost on the current fold
    model_reg.fit(
        X_train,
        y_train,
        cat_features=cat_feature,
        verbose=False
    )

    # Predict on the validation fold
    pred = model_reg.predict(X_valid)

    # Store the validation R² score
    scores.append(r2_score(y_valid, pred))
print("Fold R²:", scores)
print("Mean CV R²:", np.mean(scores))
print("Std CV R²:", np.std(scores))

model_reg.fit(
    train_x,
    train_y,
    cat_features=cat_feature,
    verbose=False
)

pred = model_reg.predict(test_x)
trainlog_pred = model_reg.predict(train_x)
model_pred = model_reg.predict(test_x)

print("rmse" , np.sqrt(mean_squared_error(np.expm1 (test_y) , np.expm1(model_pred))))
print("train r2" , r2_score( np.expm1(train_y) , np.expm1(trainlog_pred)))
print("r2test" , r2_score(np.expm1(test_y) , np.expm1(model_pred) ))
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score


feature_importance = model_reg.get_feature_importance()

importance_df = (
    pd.DataFrame({
        "Feature": train_x.columns,
        "Importance": feature_importance
    })
    .sort_values("Importance", ascending=False)
)


print(importance_df)

#* Fold R²: [0.8965465396686568, 0.8421335183668146, 0.8882830877585784, 0.875834248859861, 0.9116925838675597]
#* Mean CV R²: 0.8828979957042942
#* Std CV R²: 0.023477859392851175
#* rmse 2453876315.3600287
#* train r2 0.9037919370958578
#* r2test 0.8572842553923125
#*                  Feature  Importance
#* 0                   Area   24.734261
#* 5                Address   18.003651
#* 6        district_number   15.787192
#* 11  district_target_mean   12.060095
#* 7              is_suburb    9.160447
#* 9         facility_score    6.541213
#* 10         area_per_room    5.097734
#* 8               top_area    4.151260
#* 1                   Room    2.624013
#* 2                Parking    0.857834
#* 4               Elevator    0.619499
#* 3              Warehouse    0.362801