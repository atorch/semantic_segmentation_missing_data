from tensorflow.keras import losses, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    concatenate,
)


BASE_N_FILTERS = 64
ADDITIONAL_FILTERS_PER_BLOCK = 16
DROPOUT_RATE = 0.0


def add_downsampling_block(input_layer, block_index, n_blocks):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    batchnorm = BatchNormalization()(conv2)

    # Note: Don't MaxPool in last downsampling block
    if block_index == n_blocks - 1:

        return batchnorm, conv2

    return MaxPooling2D()(batchnorm), conv2


def add_upsampling_block(input_layer, block_index, downsampling_conv2_layers):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    upsample = UpSampling2D()(input_layer)

    concat = concatenate([upsample, downsampling_conv2_layers[block_index - 1]])

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(concat)
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    return BatchNormalization()(conv2)


def get_model(image_shape, n_blocks, n_classes):

    # Note: the model is fully convolutional: the input image width and height can be arbitrary
    input_layer = Input(shape=(None, None, image_shape[2]))

    initial_layer = Conv2D(16, kernel_size=1, activation="relu")(input_layer)

    # Note: Keep track of conv2 layers so that they can be connected to the upsampling blocks
    downsampling_conv2_layers = []

    current_last_layer = initial_layer
    for index in range(n_blocks):

        current_last_layer, conv2_layer = add_downsampling_block(
            current_last_layer, index, n_blocks
        )

        downsampling_conv2_layers.append(conv2_layer)

    for index in range(n_blocks - 1, 0, -1):

        current_last_layer = add_upsampling_block(
            current_last_layer, index, downsampling_conv2_layers
        )

    output = Conv2D(n_classes, 1, activation="softmax", name="output")(
        current_last_layer
    )

    model = Model(
        inputs=input_layer,
        outputs=[
            output,
        ],
    )

    print(model.summary())

    nadam = optimizers.Nadam()

    model.compile(
        optimizer=nadam,
        loss={
            "output": losses.categorical_crossentropy,
        },
        metrics=["accuracy"],
    )

    return model

