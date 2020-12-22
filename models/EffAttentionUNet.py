from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, BatchNormalization, Concatenate, Activation

from layers.AttentionGate import attention_gate, double_conv_layer

OUTPUT_MASK_CHANNEL = 3
# network structure
FILTER_NUM = 16  # number of basic filters for the first layer
FILTER_SIZE = 3


def find_block_ends(model):
    layer_names = [layer.name for layer in model.layers]
    prev_num = 0
    ptr_current = 1

    results = []
    wanted_num = {2, 3, 4, 6, 7}

    while ptr_current < len(layer_names):
        if layer_names[ptr_current][:5] == "block":
            block_num = int(layer_names[ptr_current][5])
            if block_num != prev_num:
                if prev_num in wanted_num:
                    results.append(layer_names[ptr_current - 1])
                prev_num = block_num
        ptr_current += 1

    results.append("block7b_add")
    return results


def eff_attention_unet(base_model, inputs, output_channel, dropout, batch_norm):
    connection_layers = find_block_ends(base_model)

    # upsampling
    block_7 = base_model.get_layer(connection_layers[-1]).output
    up_block_7 = Conv2DTranspose(FILTER_NUM * 16, FILTER_SIZE, strides=2, padding='same',
                                 kernel_initializer='he_normal')(block_7)
    block_6 = base_model.get_layer(connection_layers[-2]).output
    attention_76 = attention_gate(up_block_7, block_6, FILTER_NUM)
    cat_block_6 = Concatenate(axis=-1)([attention_76, up_block_7])
    conv_block_6 = double_conv_layer(cat_block_6, FILTER_SIZE, FILTER_NUM * 8, dropout, batch_norm)

    up_block_6 = Conv2DTranspose(FILTER_NUM * 8, FILTER_SIZE, strides=2, padding='same',
                                 kernel_initializer='he_normal')(conv_block_6)
    block_4 = base_model.get_layer(connection_layers[-3]).output
    attention_64 = attention_gate(up_block_6, block_4, 2 * FILTER_NUM)
    cat_block_4 = Concatenate(axis=-1)([attention_64, up_block_6])
    conv_block_4 = double_conv_layer(cat_block_4, FILTER_SIZE, FILTER_NUM * 4, dropout, batch_norm)
    up_block_4 = Conv2DTranspose(FILTER_NUM * 4, FILTER_SIZE, strides=2, padding='same',
                                 kernel_initializer='he_normal')(conv_block_4)
    block_3 = base_model.get_layer(connection_layers[-4]).output
    attention_43 = attention_gate(up_block_4, block_3, 4 * FILTER_NUM)
    cat_block_3 = Concatenate(axis=-1)([attention_43, up_block_4])
    conv_block_3 = double_conv_layer(cat_block_3, FILTER_SIZE, FILTER_NUM * 2, dropout, batch_norm)
    up_block_3 = Conv2DTranspose(FILTER_NUM * 2, FILTER_SIZE, strides=2, padding='same',
                                 kernel_initializer='he_normal')(conv_block_3)
    block_2 = base_model.get_layer(connection_layers[-5]).output
    attention_32 = attention_gate(up_block_3, block_2, 8 * FILTER_NUM)
    cat_block_2 = Concatenate(axis=-1)([attention_32, up_block_3])
    conv_block_2 = double_conv_layer(cat_block_2, FILTER_SIZE, FILTER_NUM, dropout, batch_norm)
    up_block_2 = Conv2DTranspose(FILTER_NUM, FILTER_SIZE, strides=2, padding='same',
                                 kernel_initializer='he_normal')(conv_block_2)

    conv_final = Conv2D(output_channel, kernel_size=(1, 1))(up_block_2)
    conv_final = BatchNormalization()(conv_final)
    conv_final = Activation('relu')(conv_final)

    model = Model(inputs, conv_final, name="EffAttentionUNet")
    return model