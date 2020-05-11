# plant-pathology-2020-fgvc7-pytorch
Solution to kaggle competition on Plant leaf classification



## Exp1 : 
- training with backbone freeze of `resnext50_32x4d` 
- submission score : 0.848 
- trained for 7 epochs only 
- on kaggle 

## Exp2 
- training with backbone freeze of `resnext50_32x4d`
- trained for about 25 epochs 
- submission score : 0.850 
- `20200502-092213`


## Exp3 
- training with complete parameter learning of `resnext50_32x4d`
- improved a lot : 0.93 score 
- `20200502-104119`

## Exp4 
- training with complete parameter learning of `resnext50_32x4d`
- loss function is non-weighted
- `20200502-113934`
- score improved : 0.941

## Exp4 
- same as 4 with more data augmentations 
- backbone: `resnext50_32dx4d`
- havent tested 

## Exp5 
- same as 4 with more data augmentations 
- backbone: `resnext50_32dx4d`
- added weighted sampler for dataset
- weights are inverse of the class 
- `20200502-143536`

## Exp6
- same as 5 
- backbone: `resnet18'`
- added weighted sampler for dataset
- weights are inverse of the class 
- increase lr reduction with patience 4 
- `20200502-163715`

## Exp7
- same as 6 
- backbone: `resnet34`
- havent tested on on submission score yet 
- `20200502-170527`

## Exp8 
- backbone: `resnext50_32dx4d`
- image size is reduced to 224 
- not submitted 
- `20200503-033216`

## Exp 9 
- backbone : `resnext50_32dx4d`
- augmentation same as exp 5 
- optimizer changed from adam to Radam 

## Exp 10
- backbone : `resnext50_32dx4d`
- augmentation same as exp 5 
- optimizer changed from adam to Radam 
- all of exp 9 
- adds weights of x-entropy loss
- `20200503-043910`

## Exp 11 
- change backbone : `efficientnet`
- slow update 
- `20200503-045711`

## Exp 12 
- same as exp 11 
- increase learning rate 
- weight decay set to `1e-4`
- `20200503=050919`


## Exp 13 
- same as exp 12 
- change RAdam to adam
- converges to max of 90% acc
- `20200503-053943`

## Exp 14 
- same as exp 13 
- remove loss weights since we are over sampling. 
- backbone : `resnet50`
- binary classification: healthy / unhealthy 
- `20200503-063152`

## Exp 15 
- same as exp 14 
- remove loss weights since we are over sampling. 
- backbone : `resnext50_32x4d`
- binary classification: healthy / unhealthy
- `20200503-074612`

## Exp 16 
- trains for 3 class classification 
- setting is similar to exp 15 
- no healthy dataset is used (train/test)
- accuracy started decreasing after few epochs

## Exp 17
- same as exp 16 but without weights for data sampling
- trains for 3 class classification 
- setting is similar to exp 15 
- no healthy dataset is used (train/test)
- accuracy started decreasing after few epochs
- `20200503-091324`
- Evaluation with both h/u and 3 class classification is not yielding anything

## Exp 18
- split dataset into 4 groups 
- trained ensemle of 4 models for each dataset 
- evaluation score: 0.95 
- `split1_`* 

## Exp 19 
- multi-class classification with BCE
- this provides individual score for each category of rust or scab 
- instead of doing 1-class prediction  
- `unhealthy_seresnet1010`