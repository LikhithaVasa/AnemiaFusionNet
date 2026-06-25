import pickle
import pandas as pd


model = pickle.load(
    open(
        "geo_model.pkl",
        "rb"
    )
)


def predict_geo_risk(input_data):

    data = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(
        data
    )

    probability = model.predict_proba(
        data
    )

    return {
        "risk": prediction[0],
        "confidence": max(probability[0])
    }