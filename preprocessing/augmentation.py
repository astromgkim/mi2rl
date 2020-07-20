import albumentations as albu

def augmentation_train(): 
    train_transform = [ albu.HorizontalFlip(p=0.5), 
                       albu.VerticalFlip(p=0.5), 
#                        albu.MultiplicativeNoise(multiplier=(0.98, 1.02), per_channel=True, p=0.4), 
#                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),contrast_limit=(-0.1, 0.1),brightness_by_max=True, p=0.5), 
                       albu.RandomGamma(gamma_limit=(80,120), p=0.5),

    albu.OneOf(
    [albu.ElasticTransform(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,alpha=1,sigma=25,alpha_affine=25, p=0.5),
     albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-0.3,0.3),num_steps=5, p=0.5),
     albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,interpolation=cv2.INTER_CUBIC,distort_limit=(-.05,.05),shift_limit=(-0.1,0.1), p=0.5),
    ]
    ,p=0.5),
            
    albu.OneOf([            
    albu.IAASharpen(alpha=(0,0.1), lightness=(0.01,0.03), p=0.3),
    albu.MotionBlur(blur_limit=(3), p=0.1),
    albu.GaussianBlur(blur_limit=(5), p=0.1),
    ],p=0.5),
    
    albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, p=0.5),
#     albu.Normalize()
# #     albu.RandomCrop(height=448, width=448, always_apply=True),
    ]
    
    return albu.Compose(train_transform)
        
def augmentation_tune(): 
    train_transform = [ 
    
#     albu.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC, p=0.5),
#     albu.RandomCrop(height=448, width=448, always_apply=True),
    ]
    
    return albu.Compose(train_transform)
        
