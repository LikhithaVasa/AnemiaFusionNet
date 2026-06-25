import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    GlobalAveragePooling1D,
    Concatenate,
    Reshape
)

from tensorflow.keras.models import Model



def create_transformer_fusion_model():


    # Inputs

    image_input = Input(
        shape=(128,),
        name="image_features"
    )


    clinical_input = Input(
        shape=(16,),
        name="clinical_features"
    )


    geo_input = Input(
        shape=(16,),
        name="geo_features"
    )



    # Feature projection

    image_proj = Dense(64, activation="relu")(
        image_input
    )


    clinical_proj = Dense(64, activation="relu")(
        clinical_input
    )


    geo_proj = Dense(64, activation="relu")(
        geo_input
    )



    # Combine modalities as tokens

    tokens = Concatenate()(
        [
            image_proj,
            clinical_proj,
            geo_proj
        ]
    )


    tokens = Reshape(
        (3,64)
    )(tokens)



    # Transformer Encoder

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(
        tokens,
        tokens
    )


    x = LayerNormalization()(
        tokens + attention
    )


    x = Dropout(0.2)(x)


    x = GlobalAveragePooling1D()(x)



    # Classification head

    x = Dense(
        64,
        activation="relu"
    )(x)


    output = Dense(
        1,
        activation="sigmoid"
    )(x)



    model = Model(
        inputs=[
            image_input,
            clinical_input,
            geo_input
        ],
        outputs=output
    )



    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    return model




if __name__ == "__main__":


    model = create_transformer_fusion_model()


    model.summary()


    model.save(
        "fusion_transformer.keras"
    )


    print(
        "Transformer Fusion Model Saved"
    )