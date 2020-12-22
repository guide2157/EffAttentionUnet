from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Multiply, Add, \
    Activation, AveragePooling2D
import tensorflow.keras.backend as K


def down_sample(inputs, input_shape):
    max_pool_1 = MaxPooling2D(strides=2)(inputs)

    conv7_1 = Conv2D(input_shape[-1], 7, padding='same', kernel_initializer='he_normal')(max_pool_1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = Activation('relu')(conv7_1)

    conv7_2 = Conv2D(input_shape[-1], 7, padding='same', kernel_initializer='he_normal')(conv7_1)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = Activation('relu')(conv7_2)

    max_pool_2 = MaxPooling2D(strides=2)(conv7_1)

    conv5_1 = Conv2D(input_shape[-1], 5, padding='same', kernel_initializer='he_normal')(max_pool_2)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)

    conv5_2 = Conv2D(input_shape[-1], 5, padding='same', kernel_initializer='he_normal')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = Activation('relu')(conv5_2)

    max_pool_3 = MaxPooling2D(strides=2)(conv5_1)

    conv3_1 = Conv2D(input_shape[-1], 3, padding='same', kernel_initializer='he_normal')(max_pool_3)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)

    conv3_2 = Conv2D(input_shape[-1], 3, padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)

    upsampled_8 = Conv2DTranspose(input_shape[-1], 2, strides=(2, 2), kernel_initializer='he_normal')(conv3_2)
    added_1 = Add()([upsampled_8, conv5_2])
    upsampled_16 = Conv2DTranspose(input_shape[-1], 2, strides=(2, 2), kernel_initializer='he_normal')(added_1)
    added_2 = Add()([upsampled_16, conv7_2])
    upsampled_32 = Conv2DTranspose(input_shape[-1], 2, strides=(2, 2), kernel_initializer='he_normal')(added_2)

    return upsampled_32


#
def global_pooling_branch(inputs, input_shape):
    global_pool = AveragePooling2D(input_shape[1])(inputs)
    conv1_2 = Conv2D(input_shape[-1], 1, padding='valid', kernel_initializer='he_normal')(global_pool)
    upsampled = Conv2DTranspose(input_shape[-1], input_shape[1], kernel_initializer='he_normal')(conv1_2)
    #
    return upsampled


def feature_pyramid_attention(inputs):
    input_shape = K.int_shape(inputs)
    down_up_conved = down_sample(inputs, input_shape)
    direct_conved = Conv2D(input_shape[-1], 1, padding='valid', kernel_initializer='he_normal')(inputs)
    gpb = global_pooling_branch(inputs, input_shape)
    multiplied = Multiply()([down_up_conved, direct_conved])
    added_fpa = Add()([multiplied, gpb])

    return added_fpa
