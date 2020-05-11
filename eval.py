# general imports
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

import cv2
import os  
from PIL import Image 
from pprint import pprint
import time
from tqdm import tqdm


# torch and torchvision
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# catalyst for training and metrics
from catalyst.utils import split_dataframe_train_test
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback
from catalyst.dl import SupervisedRunner

# scikit-learn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import timm

from dataset import LeafDataset
from utils import get_transforms
from config import config

import warnings
warnings.simplefilter("ignore")

model = None

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # set the following true for inference otherwise training speed will be slow
    torch.backends.cudnn.deterministic = True


def load_timm_model(arch, num_classes, checkpoint_location, location='cpu'):
    """defines backbone architecture and updates final layer"""
    model = timm.create_model(
        model_name=arch, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_location,
                            map_location={'cuda:0': location})
    pprint(checkpoint['epoch_metrics'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def run_evaluation(csv_path, img_root, img_transforms, device):
    global model
    # given model and valid dataset
    # iterate over dataset and compute prediction
    y_true = []
    y_pred = []
    y_logits = []
    misses = {}
    df = pd.read_csv(csv_path)
    test_size = len(df)
    print(f"Size: {test_size}")

    model.eval()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['image_id']
        target = np.argmax(
            row[['healthy', 'multiple_diseases', 'rust', 'scab']].values)

        img = Image.open(os.path.join(
            img_root, filename+'.jpg')).convert('RGB')
        img = np.asarray(img)

        augmented = img_transforms(image=img)
        img_tensor = augmented['image']
        img_tensor = img_tensor.unsqueeze(0,)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = F.softmax(model(img_tensor)).squeeze().cpu()
            probs = pred.numpy()
            _, output = torch.topk(pred, 1)
            output = output.numpy()[0]
            if output != target:
                misses[filename] = {
                    'y_true': target,
                    'y_pred': probs
                }
        y_true.append(target)
        y_pred.append(output)
        y_logits.append(probs)

    return y_true, y_pred, y_logits, misses


def read_sample(root: str, filename: str):
    img = cv2.imread(os.path.join(root, filename+'.jpg'))
    return img


def plot_misses(df, img_root, result_filename):
    fig, ax = plt.subplots(nrows=len(df), ncols=2, figsize=(10, 20))

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        #row = df.iloc[idx]
        filename = row['image_id']
        print(filename)
        true_label = row['y_true']
        pred_label = np.argmax(row['y_pred'])

        # label = np.argmax(df.ilo[['healthy', 'multiple_diseases', 'rust', 'scab']].values)
        label_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
        img = read_sample(img_root, filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax[idx][0].imshow(img)
        ax[idx][0].axis('off')
        ax[idx][0].set_title(
            f"{filename}: true: {label_names[true_label]} pred: {label_names[pred_label]}")

        sns.barplot(x=label_names, y=row['y_pred'], ax=ax[idx][1])
        ax[idx][1].set_title('Probs')
    plt.savefig(result_filename)
    return


def run_on_held_out(csv_path, img_root, img_transforms, device):
    global model
    # given model and valid dataset
    # iterate over dataset and compute prediction

    df = pd.read_csv(csv_path)
    test_size = len(df)
    print(f"Size: {test_size}")
    y_pred = {}
    model.eval()
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        filename = row['image_id']
        # target = np.argmax(row[['healthy', 'multiple_diseases', 'rust', 'scab']].values)
        img = Image.open(os.path.join(
            img_root, filename+'.jpg')).convert('RGB')
        img = np.asarray(img)

        augmented = img_transforms(image=img)
        img_tensor = augmented['image']
        img_tensor = img_tensor.unsqueeze(0,)
        img_tensor = img_tensor.to(device)

        # run prediction
        with torch.no_grad():
            pred = F.softmax(
                model(img_tensor)).squeeze().cpu()
            # _,output = torch.topk(pred,1)
            output = pred.numpy()
            result = {
                'healthy': output[0],
                'multiple_diseases': output[1],
                'rust': output[2],
                'scab': output[3]
            }
        # store results
        y_pred[filename] = result

    return y_pred


def main():
    global model
    # setup config
    cfg = config()
    cfg['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # cfg['logdir'] += timestr
    set_global_seed(cfg['random_state'])
    pprint(cfg)

    # load data
    train_df = pd.read_csv(cfg['train_csv_path'])
    test_df = pd.read_csv(cfg['test_csv_path'])
    print(len(train_df), len(test_df))
    # train_img_weights = compute_dataset_weights(train_df)

    _, test_transforms = get_transforms(cfg['input_size'])

    # model setup
    if model is None:
        checkpoint_location1 = "./logs2/tf_efficientnet_b5_ns_exp10_456_labelsmooth_adamw_split2_20200511-045005"
        model = load_timm_model(
            cfg['arch'], len(cfg['class_names']), checkpoint_location1+"/checkpoints/best.pth", location='cpu')
        model.to(cfg['device'])
        model.eval()

    print("Run on test set ...")
    test_true, test_pred, test_probs, misses = run_evaluation(
        './processed/test2.csv', cfg['img_root'], test_transforms, cfg['device'])

    misses_df = pd.DataFrame.from_dict(
        misses, orient='index', columns=['y_true', 'y_pred'])
    misses_df.index.name = 'image_id'
    misses_df = misses_df.reset_index()
    print("Number of miss:", len(misses_df))

    print(classification_report(test_true, test_pred,
                                target_names=cfg['class_names']))
    print(
        f"ROC AUC Score:{roc_auc_score(test_true, np.array(test_probs), multi_class='ovr')}")

    print("Creating CM ...")
    cm = confusion_matrix(test_true, test_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm, annot=True,)
    plt.title('Confustion Matrix for prediction')
    plt.savefig(f"./result/{cfg['arch']}_exp{cfg['exp_idx']}_cm.pdf")

    print("Plotting Misses ....")
    plot_misses(misses_df, cfg['img_root'],
                f"./result/{cfg['arch']}_exp{cfg['exp_idx']}_misses.pdf")

    print("Run on eval set ...")
    submission_dict = run_on_held_out(
        cfg['eval_csv_path'], cfg['test_img_root'], test_transforms, cfg['device'])

    print("Writing submissions...")
    submission_df = pd.DataFrame.from_dict(
        submission_dict, orient='index', columns=cfg['class_names'])
    submission_df.index.name = 'image_id'
    submission_df.to_csv(f"./result/submission_{cfg['arch']}_{cfg['exp_idx']}.csv")


if __name__ == "__main__":
    main()
