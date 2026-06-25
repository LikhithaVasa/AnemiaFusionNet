import pickle
import numpy as np


# Load trained models

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



def multimodal_prediction(
    clinical_data,
    geo_data,
    image_score
):

    # Clinical prediction
    clinical_result = clinical_model.predict(
        clinical_data
    )[0]


    # Geo prediction
    geo_result = geo_model.predict(
        geo_data
    )[0]


    # Fusion score

    final_score = (
        0.4 * clinical_result +
        0.3 * geo_result +
        0.3 * image_score
    )


    if final_score < 0.33:
        risk = "Low Risk"

    elif final_score < 0.66:
        risk = "Medium Risk"

    else:
        risk = "High Risk"


    return {
        "clinical_prediction": clinical_result,
        "geo_score": geo_result,
        "image_score": image_score,
        "final_score": final_score,
        "risk": risk
    }