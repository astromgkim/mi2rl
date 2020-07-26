import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

image_shape = (1024, 1024, 3)
def perceptual_loss(y_true, y_pred):
    
    y_true = K.concatenate( [y_true for i in range(3)], axis=-1 )
    y_pred = K.concatenate( [y_pred for i in range(3)], axis=-1 )
    y_true = K.cast( y_true, 'float32' )
    y_pred = K.cast( y_pred, 'float32' )
    
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def perceptual_mae_combined_loss(y_true, y_pred):
    
    mae_loss = K.mean(K.sum(K.abs(y_true - y_pred), axis=-1))
    
    y_true = K.concatenate( [y_true for i in range(3)], axis=-1 )
    y_pred = K.concatenate( [y_pred for i in range(3)], axis=-1 )
    y_true = K.cast( y_true, 'float32' )
    y_pred = K.cast( y_pred, 'float32' )
    
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred))) + mae_loss


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    TP = y_true * y_pred
    FN = y_true * (1-y_pred)
    return TP / (TP+FN +K.epsilon())

