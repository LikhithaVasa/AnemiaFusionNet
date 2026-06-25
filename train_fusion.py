import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



# -----------------------
# Load models
# -----------------------

image_model = load_model(
    r"D:\AnemiaFusionNet\image_modality\image_model.keras"
)


fusion_model = load_model(
    "fusion_transformer.keras"
)



clinical_model = pickle.load(
    open(
        r"D:\AnemiaFusionNet\clinical_modality\clinical_model.pkl",
        "rb"
    )
)


geo_model = pickle.load(
    open(
        r"D:\AnemiaFusionNet\geo_risk\geo_model.pkl",
        "rb"
    )
)



# -----------------------
# Clinical Data
# -----------------------

clinical_df = pd.read_excel(
    r"D:\AnemiaFusionNet\data\combined_data.xlsx"
)


labels = clinical_df["Anemia_Label"]


X_clinical = clinical_df[
    [
        "Number",
        "Hgb",
        "Gender",
        "Age",
        "Country"
    ]
]


X_clinical = X_clinical.copy()


X_clinical["Gender"] = (
    X_clinical["Gender"]
    .map({"M":0,"F":1})
)


X_clinical["Country"] = (
    X_clinical["Country"]
    .map({"India":0,"Italy":1})
)



# -----------------------
# Create dummy features
# (for connecting trained models)
# -----------------------

samples = len(X_clinical)



image_features = np.random.rand(
    samples,
    128
)


clinical_features = np.random.rand(
    samples,
    16
)


geo_features = np.random.rand(
    samples,
    16
)



# -----------------------
# Train Transformer
# -----------------------

X_train = [
    image_features,
    clinical_features,
    geo_features
]


X1,X2,y1,y2 = train_test_split(
    image_features,
    labels,
    test_size=0.2,
    random_state=42
)


history = fusion_model.fit(
    [
        image_features,
        clinical_features,
        geo_features
    ],
    labels,
    epochs=10,
    batch_size=8,
    validation_split=0.2
)



fusion_model.save(
    "final_anemia_fusion_model.keras"
)


print(
    "Final Transformer Fusion Model Saved"
)
