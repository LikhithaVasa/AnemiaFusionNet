import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv(
    r"D:\AnemiaFusionNet\data\NFHS_india_district_insights.csv"
)

print(df.head())
print(df.columns)


# Encode text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])


# Create Geo Risk Target
# Using important health indicators to calculate risk

risk_columns = [
    col for col in df.columns
    if "anemia" in col.lower()
]


if len(risk_columns) > 0:
    target = df[risk_columns].mean(axis=1)
else:
    # fallback using health-related indicators
    target = (
        df.mean(axis=1)
    )


# Remove target columns from input
X = df.drop(
    columns=risk_columns,
    errors="ignore"
)

y = target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


pred = model.predict(X_test)


print(
    "MAE:",
    mean_absolute_error(y_test,pred)
)

print(
    "R2 Score:",
    r2_score(y_test,pred)
)


with open(
    "geo_model.pkl",
    "wb"
) as f:
    pickle.dump(model,f)


print("Geo Risk Model Saved")