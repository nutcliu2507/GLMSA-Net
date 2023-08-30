import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Conv2D, Conv2DTranspose, Flatten,
                                     Input, MaxPool2D, Reshape, UpSampling2D,
                                     ZeroPadding2D, concatenate, Dropout, SeparableConv2D, DepthwiseConv2D,
                                     GlobalAveragePooling2D, Lambda, Concatenate)
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import tensorflow.keras.backend as backend
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.utils as keras_utils


def squeeze_excite_block(x, ratio=16):
    init = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = layers.Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x

def S_A_block(x):
    se_output = x
    maxpool_spatial = layers.Lambda(lambda x: backend.max(x, axis=3, keepdims=True))(se_output)
    avgpool_spatial = layers.Lambda(lambda x: backend.mean(x, axis=3, keepdims=True))(se_output)
    max_avg_pool_spatial = layers.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    SA = layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                       kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    x = layers.Multiply()([se_output, SA])

    return x

def up_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.UpSampling2D(interpolation='bilinear')(main_branch)
    main_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.UpSampling2D(interpolation='bilinear')(input_tensor)
    skip_branch = layers.Conv2D(channels // 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])

def down_sampling_module(input_tensor):
    channels = list(input_tensor.shape)[-1]
    main_branch = layers.Conv2D(channels, kernel_size=(1, 1), activation="relu")(
        input_tensor
    )
    main_branch = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(main_branch)
    main_branch = layers.MaxPooling2D()(main_branch)
    main_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(main_branch)
    skip_branch = layers.MaxPooling2D()(input_tensor)
    skip_branch = layers.Conv2D(channels * 2, kernel_size=(1, 1))(skip_branch)
    return layers.Add()([skip_branch, main_branch])

def dual_attention_unit_block(input_tensor):
    channels = list(input_tensor.shape)[-1]
    feature_map = layers.Conv2D(
        channels, kernel_size=(3, 3), padding="same", activation="relu"
    )(input_tensor)
    feature_map = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(
        feature_map
    )
    channel_attention = squeeze_excite_block(feature_map)
    spatial_attention = S_A_block(feature_map)
    concatenation = layers.Concatenate(axis=-1)([channel_attention, spatial_attention])
    concatenation = layers.Conv2D(channels, kernel_size=(1, 1))(concatenation)
    return layers.Add()([input_tensor, concatenation])

def selective_kernel_feature_fusion(multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3):
    channels = list(multi_scale_feature_1.shape)[-1]
    combined_feature = layers.Add()(
        [multi_scale_feature_1, multi_scale_feature_2, multi_scale_feature_3]
    )
    gap = layers.GlobalAveragePooling2D()(combined_feature)
    channel_wise_statistics = tf.reshape(gap, shape=(-1, 1, 1, channels))
    compact_feature_representation = layers.Conv2D(
        filters=channels // 8, kernel_size=(1, 1), activation="relu"
    )(channel_wise_statistics)
    feature_descriptor_1 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_2 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_descriptor_3 = layers.Conv2D(
        channels, kernel_size=(1, 1), activation="softmax"
    )(compact_feature_representation)
    feature_1 = multi_scale_feature_1 * feature_descriptor_1
    feature_2 = multi_scale_feature_2 * feature_descriptor_2
    feature_3 = multi_scale_feature_3 * feature_descriptor_3
    aggregated_feature = layers.Add()([feature_1, feature_2, feature_3])
    return aggregated_feature

def multi_scale_residual_block(input_tensor, channels):
    # features
    level1 = input_tensor
    level2 = down_sampling_module(input_tensor)
    level3 = down_sampling_module(level2)
    # DAU
    level1_dau = dual_attention_unit_block(level1)
    level2_dau = dual_attention_unit_block(level2)
    level3_dau = dual_attention_unit_block(level3)
    # SKFF
    level1_skff = selective_kernel_feature_fusion(
        level1_dau,
        up_sampling_module(level2_dau),
        up_sampling_module(up_sampling_module(level3_dau)),
    )
    level2_skff = selective_kernel_feature_fusion(
        down_sampling_module(level1_dau), level2_dau, up_sampling_module(level3_dau)
    )
    level3_skff = selective_kernel_feature_fusion(
        down_sampling_module(down_sampling_module(level1_dau)),
        down_sampling_module(level2_dau),
        level3_dau,
    )
    # DAU 2
    level1_dau_2 = dual_attention_unit_block(level1_skff)
    level2_dau_2 = up_sampling_module((dual_attention_unit_block(level2_skff)))
    level3_dau_2 = up_sampling_module(
        up_sampling_module(dual_attention_unit_block(level3_skff))
    )
    # SKFF 2
    skff_ = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)
    conv = layers.Conv2D(channels, kernel_size=(3, 3), padding="same")(skff_)
    return layers.Add()([input_tensor, conv])

# Decoder for UNet is adapted from keras-segmentation
def UNet(f4, f3, f2, f1, output_height, output_width, l1_skip_conn=True, n_classes=6):
    # ----------------------fcn----------------------------------------------------
    x = SeparableConv2D(filters=4096, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu',
                        kernel_regularizer=regularizers.L2(0))(f4)
    x = Dropout(rate=0.5)(x)
    x = SeparableConv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                        kernel_regularizer=regularizers.L2(0))(x)
    fc_output = Dropout(rate=0.5)(x)
    o = f4

    IMAGE_ORDERING = 'channels_last'
    if IMAGE_ORDERING == 'channels_first':
        MERGE_AXIS = 1
    elif IMAGE_ORDERING == 'channels_last':
        MERGE_AXIS = -1
    # ----------------------FCN--------------------------------------------------
    fc_output = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(fc_output)
    fc_output = (Conv2D(2048, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(fc_output)
    fc_output = (BatchNormalization())(fc_output)
    o = (concatenate([fc_output, o], axis=MERGE_AXIS))
    o = Dropout(rate=0.5)(o)
    o = Conv2D(1024, (1, 1), padding='same', use_bias=False)(o)
    o = squeeze_excite_block(o)
    o = S_A_block(o)

    #    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    #    o = (concatenate([o, f4], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    #    o = Dropout(0.3)(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((4, 4), data_format=IMAGE_ORDERING, interpolation='bilinear'))(o)
    o = Conv2D(n_classes, 1, activation="softmax")(o)


    return o
