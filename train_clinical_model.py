import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Load data
df = pd.read_excel(
    r"D:\AnemiaFusionNet\data\combined_data.xlsx"
)

print(df.head())
print(df.columns)


# Convert text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))


# Create target
# If anemia column exists use it
target_columns = [
    col for col in df.columns
    if "anemia" in col.lower()
]


if len(target_columns) > 0:

    y = df[target_columns[0]]

    X = df.drop(
        columns=[target_columns[0]]
    )

else:

    # fallback:
    # create risk label from numeric values

    score = df.mean(axis=1)

    y = pd.qcut(
        score,
        q=2,
        labels=[0,1]
    )

    X = df



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


pred = model.predict(
    X_test
)


print(
    "Clinical Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)


with open(
    "clinical_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


print(
    "Clinical Model Saved"
)