import numpy as np
from PIL import Image
import cv2

def percentile_xray8bit(file, rescale_size=(512, 512), output_channel=1, interpolation_method=cv2.INTER_CUBIC):
  
    img_rows = rescale_size[0]
    img_cols = rescale_size[1]
    
    tmp_img=cv2.imread(file, 0)#read raw image
    _rows, _cols = tmp_img.shape#get image size

    #resize image having img_rows x img_cols using linear interpolation method (there exists other methods of course)
    tmp_img = cv2.resize(tmp_img, (img_rows, img_cols), interpolation = interpolation_method) # interpolation
    tmp_img = tmp_img.astype('float32')#change data type
    
    #normalization and cut 1% outliers
    tmp_img -= np.min(tmp_img)
    tmp_img /= np.percentile(tmp_img, 99)
    tmp_img[tmp_img > 1] = 1.
    tmp_img *= (2**8-1)
    tmp_img = tmp_img.astype(np.uint8)
    
    #stacking gray channel image to make 3channels.
    if output_channel == 3:
      tmp_img = np.stack((tmp_img,)*3, axis=-1)
      
    return tmp_img#return preprocessed image
  

