import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
import json
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    # print('train_cls_ids', len(train_cls_ids))
    # print('val_cls_ids', len(val_cls_ids))
    # print('test_cls_ids', len(test_cls_ids))
    return train_cls_ids, val_cls_ids, test_cls_ids

def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths, labels, idx_to_class,
                 transform=None, target_transform=None, class_ids=None,
                 loader=default_loader):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        self.class_ids = class_ids

    def __getitem__(self, index):
        impath, target = self.samples[index]
        # print(len(target))
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform([target])
        # print(target)
        target = self.get_targets_multi_label(np.array(target))
        if self.class_ids is not None:
            target = target[self.class_ids]
        # print(target)
        return img, target

    def __len__(self):
        return len(self.samples)

    def get_targets_multi_label(self, target):
        # Full (non-partial) labels
        labels = np.zeros(len(self.classes))
        labels[target] = 1
        target = labels.astype('float32')
        return target

def parse_csv_data(dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "nus_wide_data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "nus_wide_data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    return images_path_list, image_labels_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, train_idx, valid_idx = parse_csv_data(dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train,
                               idx_to_class,
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, idx_to_class,
                             transform=val_transform, class_ids=train_cls_ids)
    

    return train_dl, val_dl, train_cls_ids, test_cls_ids









if __name__ == '__main__':
    dataset_local_path = '/media/data/maleilei/MLICdataset/NUS_WIDE/nuswide'
    metadata_local_path = '/media/data/maleilei/MLICdataset/NUS_WIDE'
    train_transform = None
    val_transform = None
    json_path = '/media/data/maleilei/MLICdataset/NUS_WIDE/benchmark_81_v0.json'

    import pandas as pd
    df = pd.read_csv(os.path.join(metadata_local_path, "nus_wide_data.csv")) # 169823
    # df = pd.read_csv(os.path.join(metadata_local_path, "zsl_nus_wide_data.csv")) # 216546
    # len('/media/data/maleilei/MLICdataset/NUS_WIDE/images') # 223496
    images_path_list = df.values[:, 0]
    print(len(images_path_list))





    # get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
    #                       val_transform, json_path)