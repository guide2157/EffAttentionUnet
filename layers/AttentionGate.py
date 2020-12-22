from tensorflow.keras.layers import Conv2D, Multiply, Add, Conv2DTranspose, Activation, UpSampling2D, Dropout, \
    BatchNormalization
import tensorflow.keras.backend as K

"""
Code provided by MoleImg
"""


def attention_gate(x, g, inter_shape):
    shape_g = K.int_shape(g)
    shape_x = K.int_shape(x)
    conv_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    shape_theta_x = K.int_shape(conv_x)

    conv_g = Conv2D(inter_shape, (1, 1), padding='same', kernel_initializer='he_normal')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same',
                                 kernel_initializer='he_normal')(conv_g)
    concat_xg = Add()([upsample_g, conv_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', kernel_initializer='he_normal')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
        sigmoid_xg)

    # upsample_psi = Lambda(lambda x, repnum: tf.repeat(x, repnum, axis=3),
    #                       arguments={'repnum': shape_x[-1]})(upsample_psi, shape_x[3])

    y = Multiply()([upsample_psi, x])

    return y


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    axis = 3
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=axis)(shortcut)

    res_path = Add()([shortcut, conv])
    return res_path
