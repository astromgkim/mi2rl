

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.applications.vgg19 import VGG19

def plv(y_true, y_pred):
    # must ensure that y_true and y_pred come from same distribution
    # nice to have - similar distribution to vgg19 inputs
    y_pred = tf.image.grayscale_to_rgb(y_pred, name=None) #need to convert as UNet outputs single channel
    vgg = VGG19(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
    loss_model = Model(
        inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output
    )
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))



def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    TP = y_true * y_pred
    FN = y_true * (1-y_pred)
    return TP / (TP+FN +K.epsilon())
