import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, LeakyReLU, Conv2D,
                                     Dense, Reshape, SeparableConv2D, concatenate, BatchNormalization,
                                     GlobalAveragePooling2D, Lambda, Concatenate, Permute, multiply)

from group_norm import GroupNormalization

def MSA(tensor):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]
    newtensor = Conv2D(filters=filters/2, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    filters2 = newtensor.shape[channel_axis]

    y_CA = squeeze_excite_block(newtensor)

    y_1 = SeparableConv2D(filters=filters2, kernel_size=1, dilation_rate=1, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=6, padding='same',
                          kernel_initializer='he_normal', use_bias=False)(tensor)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=12, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(tensor)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=18, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(tensor)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)


    y_24 = SeparableConv2D(filters=filters2, kernel_size=3, dilation_rate=24, padding='same',
                           kernel_initializer='he_normal', use_bias=False)(tensor)
    y_24 = BatchNormalization()(y_24)
    y_24 = Activation('relu')(y_24)

    y = concatenate([y_CA, y_1, y_6, y_12, y_18, y_24])

    y = Conv2D(filters=filters, kernel_size=3, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def GLA(o):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    o1 = Conv2D(filters // 2, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(o)
    g_o = gobal_Attention(o1)
    l_o = local_Attention(o1)
    o = Concatenate()([g_o, l_o])
    o1 = ConvLeakyRelu(filters, 3)(o)
    o = Add()([o1, o])
    return o

def ws_reg(kernel):
  kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
  kernel = kernel - kernel_mean
  kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
  kernel = kernel / (kernel_std + 1e-5)

def squeeze_excite_block(x, ratio=16):
    init = x
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)


    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def gobal_Attention(o):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    tensor_input = o
    dot_f = dot_p_a(tensor_input)
    se_f = squeeze_excite_block(tensor_input)
    o = Concatenate()([se_f, dot_f])
    o = ConvLeakyRelu(filters, 3)(o)
    o = Add()([o, tensor_input])
    return o

def local_Attention(o):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = o.shape[channel_axis]
    tensor_input = o
    # Split
    o1, o2 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(o)
    o11, o12 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o1)
    o21, o22 = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(o2)

    # spatial
    os = tf.concat([o11, o12, o21, o22], axis=1)
    os = dot_p_a(os)
    os11, os12, os21, os22 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4})(os)
    # channel
    oc = squeeze_excite_block(os)
    oc11, oc12, oc21, oc22 = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 4})(oc)

    o11 = Concatenate()([oc11, os11])
    o11 = local_ConvLeakyRelu(filters, 3)(o11)
    o12 = Concatenate()([oc12, os12])
    o12 = local_ConvLeakyRelu(filters, 3)(o12)
    o21 = Concatenate()([oc21, os21])
    o21 = local_ConvLeakyRelu(filters, 3)(o21)
    o22 = Concatenate()([oc22, os22])
    o22 = local_ConvLeakyRelu(filters, 3)(o22)

    # Concat 4 -> 1
    o1 = tf.concat([o11, o12], axis=2)
    o2 = tf.concat([o21, o22], axis=2)
    o = tf.concat([o1, o2], axis=1)

    o = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=ws_reg)(o)

    o = GroupNormalization(groups=32, axis=-1)(o)
    o = LeakyReLU()(o)
    o = Add()([o, tensor_input])
    return o

# -----------------------------------------------------------
def dot_p_a(inputs):
    gamma = tf.compat.v1.get_variable("gamma", shape=(1,),
                                      initializer=tf.zeros_initializer(),
                                      regularizer=None,
                                      constraint=None)
    input_shape = inputs.get_shape().as_list()
    _, h, w, filters = input_shape

    q = Conv2D(filters // 8, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    k = Conv2D(filters // 8, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    v = Conv2D(filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    vec_k = K.reshape(k, (-1, h * w, filters // 8))
    vec_qT = K.permute_dimensions(K.reshape(q, (-1, h * w, filters // 8)), (0, 2, 1))
    kqT = K.batch_dot(vec_k, vec_qT)
    softmax_kqT = Activation('softmax')(kqT)
    vec_v = K.reshape(v, (-1, h * w, filters))
    kqTv = K.batch_dot(softmax_kqT, vec_v)
    kqTv = K.reshape(kqTv, (-1, h, w, filters))
    out = gamma * kqTv + inputs
    return out

def ConvLeakyRelu(filters, kernel_size):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", kernel_regularizer=ws_reg)(x)
        x = GroupNormalization(groups=32, axis=-1)(x)
        x = LeakyReLU()(x)
        return x

    return layer

def local_ConvLeakyRelu(filters, kernel_size):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", kernel_regularizer=ws_reg)(x)
        x = GroupNormalization(groups=32, axis=-1)(x)
        x = LeakyReLU()(x)
        return x

    return layer

