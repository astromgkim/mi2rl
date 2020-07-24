
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def unet(inputs):
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    return conv10




def unet_deep_bottleneck_dilation(inputs):
    
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv0 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    up1 = UpSampling2D(size = (2,2))(conv4)
    
    merge1 = concatenate([conv3,up1], axis = 3)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(merge1)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(conv5)
    
    up2 = UpSampling2D(size = (2,2))(conv5)
    merge2 = concatenate([conv2,up2])
    
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up3 = UpSampling2D(size = (2,2))(conv6)
    merge3 = concatenate([conv1,up3])
    
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up4 = UpSampling2D(size = (2,2))(conv7)
    merge4 = concatenate([conv0,up4])
    
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    conv9 = Conv2D(1, 1, activation = 'tanh')(conv8)

    return conv9




def unet_deep16_bottleneck_dilation(inputs):
    
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up1 = UpSampling2D(size = (2,2))(conv5)
    
    merge1 = concatenate([conv4,up1], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(merge1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(conv6)
    
    up2 = UpSampling2D(size = (2,2))(conv6)
    merge2 = concatenate([conv3,up2])
    
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up3 = UpSampling2D(size = (2,2))(conv7)
    merge3 = concatenate([conv2,up3])
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up4 = UpSampling2D(size = (2,2))(conv8)
    merge4 = concatenate([conv1,up4])
    
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up5 = UpSampling2D(size = (2,2))(conv9)
    merge5 = concatenate([conv0,up5])
    
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    conv11 = Conv2D(1, 1, activation = 'tanh')(conv10)

    return conv11



def unet_deep16_bottleneck_dilation_sigmoid(inputs):
    
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv0 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool0)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up1 = UpSampling2D(size = (2,2))(conv5)
    
    merge1 = concatenate([conv4,up1], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(merge1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'he_normal')(conv6)
    
    up2 = UpSampling2D(size = (2,2))(conv6)
    merge2 = concatenate([conv3,up2])
    
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up3 = UpSampling2D(size = (2,2))(conv7)
    merge3 = concatenate([conv2,up3])
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up4 = UpSampling2D(size = (2,2))(conv8)
    merge4 = concatenate([conv1,up4])
    
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up5 = UpSampling2D(size = (2,2))(conv9)
    merge5 = concatenate([conv0,up5])
    
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv10 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    return conv11



def aggregate(l1, l2, l3, l4, l5):
    out = concatenate([l1, l2, l3, l4, l5], axis = -1)
    out =  Conv2D(320, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    return out

base_channel = 64
def unet3plus(inputs, conv_num = base_channel):
    
    XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1)
    XE1_pool = MaxPooling2D(pool_size=(2, 2))(XE1)
    XE2 = Conv2D(conv_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1_pool)
    XE2 = Conv2D(conv_num*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2)
    XE2_pool = MaxPooling2D(pool_size=(2, 2))(XE2)
    XE3 = Conv2D(conv_num*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2_pool)
    XE3 = Conv2D(conv_num*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3)
    XE3_pool = MaxPooling2D(pool_size=(2, 2))(XE3)
    XE4 = Conv2D(conv_num*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3_pool)
    XE4 = Conv2D(conv_num*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4)
    XE4 = Dropout(0.5)(XE4)
    XE4_pool = MaxPooling2D(pool_size=(2, 2))(XE4)
    XE5 = Conv2D(conv_num*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4_pool)
    XE5 = Conv2D(conv_num*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE5)
    XE5 = Dropout(0.5)(XE5)
    
    
    XD4_from_XE5 = UpSampling2D(size=(2,2), interpolation='bilinear')(XE5)
    XD4_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD4_from_XE5)
    XD4_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE4)
    XD4_from_XE3 = MaxPooling2D(pool_size=(2,2))(XE3)
    XD4_from_XE3 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE3)
    XD4_from_XE2 = MaxPooling2D(pool_size=(4,4))(XE2)
    XD4_from_XE2 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE2)
    XD4_from_XE1 = MaxPooling2D(pool_size=(8,8))(XE1)
    XD4_from_XE1 = Conv2D(conv_num, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(XD4_from_XE1)
    XD4 = aggregate(XD4_from_XE5, XD4_from_XE4, XD4_from_XE3, XD4_from_XE2, XD4_from_XE1)
    XD3_from_XE5 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE5)
    XD3_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE5)
    XD3_from_XD4 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD4)
    XD3_from_XD4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XD4)
    XD3_from_XE3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE3)
    XD3_from_XE2 = MaxPooling2D(pool_size=(2,2))(XE2)
    XD3_from_XE2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE2)
    XD3_from_XE1 = MaxPooling2D(pool_size=(4,4))(XE1)
    XD3_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD3_from_XE1)
    XD3 = aggregate(XD3_from_XE5, XD3_from_XD4, XD3_from_XE3, XD3_from_XE2, XD3_from_XE1)
    XD2_from_XE5 = UpSampling2D(size=(8,8), interpolation='bilinear')(XE5)
    XD2_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE5)
    XD2_from_XE4 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE4)
    XD2_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE4)
    XD2_from_XD3 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD3)
    XD2_from_XD3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XD3)
    XD2_from_XE2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE2)
    XD2_from_XE1 = MaxPooling2D(pool_size=(2,2))(XE1)
    XD2_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD2_from_XE1)
    XD2 = aggregate(XD2_from_XE5, XD2_from_XE4, XD2_from_XD3, XD2_from_XE2, XD2_from_XE1)
    XD1_from_XE5 = UpSampling2D(size=(16,16), interpolation='bilinear')(XE5)
    XD1_from_XE5 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE5)
    XD1_from_XE4 = UpSampling2D(size=(8,8), interpolation='bilinear')(XE4)
    XD1_from_XE4 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE4)
    XD1_from_XE3 = UpSampling2D(size=(4,4), interpolation='bilinear')(XE3)
    XD1_from_XE3 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XE3)
    XD1_from_XD2 = UpSampling2D(size=(2,2), interpolation='bilinear')(XD2)
    XD1_from_XD2 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XD1_from_XD2)
    XD1_from_XE1 = Conv2D(conv_num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(XE1)
    XD1 = aggregate(XD1_from_XE5, XD1_from_XE4, XD1_from_XE3, XD1_from_XD2, XD1_from_XE1)
    out = Conv2D(conv_num*5, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(XD1)
    out = Conv2D(1, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(out)
        
    return out
