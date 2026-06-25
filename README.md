AnemiaFusionNet
A Multimodal Feature Fusion Framework for Region-Aware Anemia Detection.

Project Overview
AnemiaFusionNet integrates:
Conjunctiva Eye Images
Clinical Data
Geographical Risk Information
using a Transformer-based Fusion Network to predict anemia risk.

Modules
Image Modality
CNN-based image feature extraction from conjunctiva eye images.
Clinical Modality

Machine learning model using:
Hemoglobin
Age
Gender
Country
Geo-Risk Module

NFHS-5 district-level health indicators used to generate regional risk scores.

Multimodal Fusion

Transformer architecture combines image, clinical, and geo features for final prediction.

Technologies
Python
TensorFlow
Keras
Scikit-Learn
Pandas
Streamlit
Results
Clinical Accuracy: 100%
Geo-Risk MAE: 3.38
Geo-Risk R² Score: 0.966
Fusion Validation Accuracy: ~75%

Author
Vasa Likhitha
