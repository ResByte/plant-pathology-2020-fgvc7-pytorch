def config():
    cfg = {
        # raw data
        'train_csv_path': './processed/train3.csv',
        'test_csv_path': './processed/test3.csv',
        'eval_csv_path': './data/test.csv',
        'img_root': './data/images/',
        'test_img_root': './data/images/',
        # dataset 
        'sampling':'weighted', # other: 'normal'
        'batch_size': 8,
        'test_batch_size': 2,
        'num_classes': 4,
        'test_size': 0.2,
        'input_size': 456,
        'class_names': ['healthy', 'multiple_diseases', 'rust', 'scab'],
        # training parameters
        'random_state': 123,
        'freeze': False,
        'logdir': './logs2/',
        'device': None,
        'num_epochs': 75,
        'exp_idx':11,
        'data_split':3,
        # optimizer
        'criterion':'label_smooth', # other : cross_entropy
        'optimizer':'adamw', # other : adam, radam
        'lr': 3e-4,
        'wd': 1e-5,
        'lr_schedule':'reduce_plateau', # cyclic_lr , 
         # backend architecture
        'arch': 'tf_efficientnet_b5_ns',# 'tf_efficientnet_b5_ns',  # 'resnet50',
        # logging
        'verbose': True,
        'check': False,  # set this true to run for 3 epochs only
    }
    return cfg