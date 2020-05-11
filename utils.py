import albumentations as albu
from albumentations.pytorch import ToTensor

def pre_transforms(image_size=512):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=2)
    ]
    
    return result

def hard_transforms():
    result = [
        # random flip 
        albu.RandomRotate90(),
        # Random shifts, stretches and turns with a 50% probability
        albu.ShiftScaleRotate( 
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        # add random brightness and contrast, 30% prob
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        # Random gamma changes with a 30% probability
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        # Randomly changes the hue, saturation, and color value of the input image 
        albu.HueSaturationValue(p=0.3),
        albu.JpegCompression(quality_lower=80),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=0.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        
    ]
    
    return result

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]

def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ])
    return result

def get_transforms(image_size):
    
    train_transforms = compose([
                        pre_transforms(image_size), 
                        hard_transforms(), 
                        post_transforms()
    ])
    
    val_transforms = compose([pre_transforms(image_size), post_transforms()])
    
    return train_transforms, val_transforms