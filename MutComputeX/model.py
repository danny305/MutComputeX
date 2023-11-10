
from tensorflow.keras import activations
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    BatchNormalization,
    Conv3D,
    Activation,
    Add,
    MaxPool3D,
)
from tensorflow.keras.regularizers import l2


def res_identity(x, f1, f2, base_name, data_format="channels_first"):

    x_skip = x

    x = Conv3D(
        filters=f1,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-identity-one-" + base_name,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv3D(
        filters=f1,
        kernel_size=(3, 3, 3),
        kernel_regularizer=l2(0.001),
        padding="same",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-identity-two-" + base_name,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv3D(
        filters=f2,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-identity-three-" + base_name,
    )(x)
    x = BatchNormalization()(x)

    x_skip = Conv3D(
        filters=f2,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-identity-skip-" + base_name,
    )(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def res_conv(x, s, f1, f2, base_name, data_format="channels_first"):

    x_skip = x

    x = Conv3D(
        filters=f1,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(s, s, s),
        data_format=data_format,
        name="Conv-redux-one-" + base_name,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv3D(
        filters=f1,
        kernel_size=(3, 3, 3),
        kernel_regularizer=l2(0.001),
        padding="same",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-redux-two-" + base_name,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv3D(
        filters=f2,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(1, 1, 1),
        data_format=data_format,
        name="Conv-redux-three-" + base_name,
    )(x)
    x = BatchNormalization()(x)

    x_skip = Conv3D(
        filters=f2,
        kernel_size=(1, 1, 1),
        kernel_regularizer=l2(0.001),
        padding="valid",
        strides=(s, s, s),
        data_format=data_format,
        name="Conv-redux-skip-" + base_name,
    )(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def create_resnet_model(input_shape, data_format="channels_first"):

    m_input = Input(shape=input_shape)

    x = BatchNormalization(axis=[1, 2, 3, 4])(m_input)

    x = res_identity(x, 50, 50, "l1-1", data_format)
    x = res_identity(x, 50, 50, "l1-2", data_format)
    x = res_identity(x, 50, 50, "l1-3", data_format)

    x = res_conv(x, 2, 50, 100, "l2-1", data_format)
    x = res_identity(x, 100, 100, "l2-2", data_format)
    x = res_identity(x, 100, 100, "l2-3", data_format)

    x = res_conv(x, 2, 100, 200, "l3-1", data_format)
    x = res_identity(x, 200, 200, "l3-2", data_format)
    x = res_identity(x, 200, 200, "l3-3", data_format)

    x = res_conv(x, 2, 200, 400, "l4-1", data_format)
    x = res_identity(x, 400, 400, "l4-2", data_format)
    x = res_identity(x, 400, 400, "l4-3", data_format)

    x = MaxPool3D(data_format=data_format)(x)

    x = Flatten()(x)
    x = Dense(1000, kernel_initializer=HeNormal())(x)
    x = Activation(activations.relu)(x)
    x = Dense(20)(x)
    x = Activation(activations.softmax)(x)

    model = Model(inputs=m_input, outputs=x, name="MutComputeX")

    return model
