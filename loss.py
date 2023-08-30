import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import dill


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)


def dice_plus_focal(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_coef(y_true, y_pred)


def _CE(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    CE_loss = - y_true[...] * K.log(y_pred)
    CE_loss = K.mean(K.sum(CE_loss, axis=-1))
    return CE_loss


def focal_loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    CE_loss = - y_true[...] * K.log(y_pred)
    CE_loss = alpha * K.pow(1 - y_pred, gamma) * CE_loss
    CE_loss = K.mean(K.sum(CE_loss, axis=-1))
    return CE_loss


class modelLoss():
    def __init__(self, lambda_, alpha, width, height, batchsize):
        self.lambda_ = lambda_
        self.width = width
        self.height = height
        self.batchsize = batchsize
        self.alpha = alpha

    def test(self, y_true, y_pred):
        # rename and split values
        # [batch, width, height, channel]
        img = y_true[:, :, :, 0:3]
        seg = y_true[:, :, :, 3:6]

        disp0 = K.expand_dims(y_pred[:, :, :, 0], -1)
        disp1 = K.expand_dims(y_pred[:, :, :, 1], -1)
        disp2 = K.expand_dims(y_pred[:, :, :, 2], -1)
        disp3 = K.expand_dims(y_pred[:, :, :, 3], -1)

        return None

    def applyLoss(self, y_true, y_pred):
        return _CE(y_true, y_pred)


    def focal_loss(self, y_true, y_pred):
        return focal_loss(y_true, y_pred)

    def dc_loss(self, y_true, y_pred):
        return dice_coef_loss(y_true, y_pred)

    def dice_plus_focal(self, y_true, y_pred):
        return dice_plus_focal(y_true, y_pred)
