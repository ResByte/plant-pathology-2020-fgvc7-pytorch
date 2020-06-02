# plant-pathology-2020-fgvc7-pytorch
This is a solution to kaggle competition on categorizing foliar disease in apple tree leaves. 

## Results 
In private leaderboard, an ensemble solution achieved 0.967 score with top 26%.   

## About Competition
Develop a model that uses images of plant leaves and categorizes them into different diseased categories. 
For further info about dataset: https://arxiv.org/abs/2004.11958 

The prediction categories as as follows: 
- `health` 
- `rust` 
- `scab` 
- `multiple` : this  conists of both rust and scab diseases

The data was highly imbalanced 

## Solution 
In order to train the model, first have a look at `config.py` and modify variables accordingly. 

### Key things: 
- Weighted sampling of data for data imbalance. 
- Variety of data random data augmentation strategies to make robust mdoel. 
- Use of **Label Smoothing** to address better predictions: https://arxiv.org/abs/1512.00567 
- **Efficientnet** as feature extractor : https://arxiv.org/abs/1905.11946 
- Pre-trained models using **noisy-student** method: https://arxiv.org/abs/1911.04252
- Optimization of cost function is based on modified version of Adam : https://arxiv.org/abs/1711.05101   

To use or not use can be configured in `config.py`:

```python 
        'criterion':'label_smooth', # other : cross_entropy
        'optimizer':'adamw', # other : adam, radam
        'lr': 3e-4, # learning rate 
        'wd': 1e-5, # weight decay parameter
        'lr_schedule':'reduce_plateau', # cyclic_lr , 
        'arch': 'tf_efficientnet_b5_ns',  # backend architecture
```

## Requirements 
- torch : `1.5.0a0+8f84ded`
- torchvision : `0.6.0a0` 

others are in `requirements.txt` 


## Improvements 

- largest model that I could use was `tf_efficientnet_b5_ns` due to limited GPU memory, there can be further improvements using `efficinet-b6, b7 or b8`. 
- training is done in straight forward manner, but improvements can be made using noisy student training or utilizing self-supervised.  

# Acknowledgements
- pytorch pre-trained image models : https://github.com/rwightman/pytorch-image-models 
- data augmentations : https://albumentations.readthedocs.io/en/latest/ 
- logging and utils : https://github.com/catalyst-team/catalyst 