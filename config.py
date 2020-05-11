def config():
    cfg = {
        # raw csv data
        'train_csv_path': './processed/train2.csv',
        'test_csv_path': './processed/test2.csv',
        # images path
        'img_root': './data/images/',
        'test_img_root': './data/images/',
        # backend architecture, features are extracted from this
        'arch': 'ssl_resnext101_32x4d',# 'tf_efficientnet_b5_ns',  # 'resnet50',
        # training parameters
        'random_state': 123,
        'num_classes': 4,
        'test_size': 0.2,
        'input_size': 600,
        'freeze': False,
        'lr': 3e-4,
        'logdir': './logs2/ssl_resnext101_32x4d_exp9_224_labelsmooth_adamw_split2_',
        'device': None,
        'batch_size': 8,
        'test_batch_size': 2,
        'num_epochs': 50,
        # logging
        'verbose': True,
        'check': False,  # set this true to run for 3 epochs only
        # data labels
        'class_names': ['healthy', 'multiple_diseases', 'rust', 'scab']

    }
    return cfg