import os
import numpy as np
import pandas as pd
import time
from pprint import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, BatchSampler
from catalyst.utils import split_dataframe_train_test
from catalyst.dl.callbacks import AccuracyCallback
from catalyst.dl import SupervisedRunner
from catalyst.contrib.nn.optimizers import RAdam
from model import Model
from dataset import LeafDataset
from utils import get_transforms
from config import config
import timm
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.optim import AdamW
import warnings
warnings.simplefilter("ignore")


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # set the following true for inference otherwise training speed will be slow
    torch.backends.cudnn.deterministic = True


def compute_dataset_weights(df):
    # ['healthy', 'multiple_diseases', 'rust', 'scab']
    # these weights are found from raw data distribution
    # inverse of frequence of each dataset
    weights = 1. / \
        torch.Tensor([0.22624931, 0.03953871, 0.27292696, 0.25974739]).double()
    print(weights)
    image_list = []
    for idx, row in df.iterrows():
        label = np.argmax(
            row[['healthy', 'multiple_diseases', 'rust', 'scab']].values)
        image_list.append(weights[label])
    return image_list


def main():
    # setup config
    cfg = config()
    cfg['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cfg['logdir'] += f"{cfg['arch']}_"
    cfg['logdir'] += f"{cfg['exp_idx']}_"
    cfg['logdir'] += f"{cfg['input_size']}_"
    cfg['logdir'] += f"{cfg['criterion']}_"
    cfg['logdir'] += f"{cfg['optimizer']}_"
    cfg['logdir'] += f"split{cfg['data_split']}_"
    cfg['logdir'] += timestr
    set_global_seed(cfg['random_state'])
    pprint(cfg)

    # load data
    train_df = pd.read_csv(cfg['train_csv_path'])
    test_df = pd.read_csv(cfg['test_csv_path'])
    print(len(train_df), len(test_df))
    train_img_weights = compute_dataset_weights(train_df)

    train_transforms, test_transforms = get_transforms(cfg['input_size'])
    train_dataset = LeafDataset(
        img_root=cfg['img_root'],
        df=train_df,
        img_transforms=train_transforms,
        is_train=True,
    )

    test_dataset = LeafDataset(
        img_root=cfg['img_root'],
        df=test_df,
        img_transforms=test_transforms,
        is_train=False,
    )
    print(
        f"Training set size:{len(train_dataset)}, Test set size:{len(test_dataset)}")

    # prepare train and test loader
    if cfg['sampling'] == 'weighted':
        # image weight based on statistics
        train_img_weights = compute_dataset_weights(train_df)
        # weighted sampler
        weighted_sampler = WeightedRandomSampler(
            weights=train_img_weights, num_samples=len(train_img_weights), replacement=False)
        # batch sampler from weigted sampler
        batch_sampler = BatchSampler(
            weighted_sampler, batch_size=cfg['batch_size'], drop_last=True)
        # train loader
        train_loader = DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=4)
    elif cfg['sampling'] == 'normal':
        train_loader = DataLoader(
            train_dataset, cfg['batch_size'], shuffle=True, num_workers=2)

    test_loader = DataLoader(
        test_dataset, cfg['test_batch_size'], shuffle=False, num_workers=1, drop_last=True)

    loaders = {
        'train': train_loader,
        'valid': test_loader
    }

    # model setup
    model = timm.create_model(model_name=cfg['arch'], num_classes=len(
        cfg['class_names']), drop_rate=0.5, pretrained=True)
    model.train()

    # loss
    if cfg['criterion'] == 'label_smooth':
        criterion = LabelSmoothingCrossEntropy()
    elif cfg['criterion'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if cfg['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    elif cfg['optimizer'] == 'adamw':
        optimizer = AdamW(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    elif cfg['optimizer'] == 'radam':
        optimizer = RAdam(
            model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    # learning schedule
    if cfg['lr_schedule'] == 'reduce_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=4)

    # trainer
    runner = SupervisedRunner(device=cfg['device'])
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,

        callbacks=[
            AccuracyCallback(
                num_classes=len(cfg['class_names']),
                threshold=0.5,
                activation="Softmax"
            ),
        ],
        logdir=cfg['logdir'],
        num_epochs=cfg['num_epochs'],
        verbose=cfg['verbose'],
        # set this true to run for 3 epochs only
        check=cfg['check'],
    )


if __name__ == "__main__":
    main()
