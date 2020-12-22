from tensorflow.keras.layers import Conv2D, Multiply, Add, Conv2DTranspose, AveragePooling2D
import tensorflow.keras.backend as K


def global_attention_upsample(high_feat, low_feat):
    high_feat_shape = K.int_shape(high_feat)
    low_feat_shape = K.int_shape(low_feat)
    global_pool = AveragePooling2D(high_feat_shape[1])(high_feat)
    conv1_1 = Conv2D(low_feat_shape[-1], 1, padding='same', kernel_initializer='he_normal')(global_pool)
    conv3_1 = Conv2D(low_feat_shape[-1], 3, padding='same', kernel_initializer='he_normal')(low_feat)
    multiplied_1 = Multiply()([conv3_1, conv1_1])
    upsampled_1 = Conv2DTranspose(low_feat_shape[-1], 2, strides=(2, 2), kernel_initializer='he_normal')(
        high_feat)
    added_1 = Add()([upsampled_1, multiplied_1])
    return added_1

