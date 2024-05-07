from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def get_model():
    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in base_model.layers:
        layer.trainable = False

    return model
