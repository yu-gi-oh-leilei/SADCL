import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import random
# import zipfile38 as zipfile
import pickle
# import shutil

object_categories = [
    'airport', 'animal', 'beach', 'bear', 'birds', 'boats', 'book', 'bridge',
    'buildings', 'cars', 'castle', 'cat', 'cityscape', 'clouds', 'computer',
    'coral', 'cow', 'dancing', 'dog', 'earthquake', 'elk', 'fire', 'fish',
    'flags', 'flowers', 'food', 'fox', 'frost', 'garden', 'glacier', 'grass',
    'harbor', 'horses', 'house', 'lake', 'leaf', 'map', 'military', 'moon',
    'mountain', 'nighttime', 'ocean', 'person', 'plane', 'plants', 'police',
    'protest', 'railroad', 'rainbow', 'reflection', 'road', 'rocks', 'running',
    'sand', 'sign', 'sky', 'snow', 'soccer', 'sports', 'statue', 'street',
    'sun', 'sunset', 'surf', 'swimmers', 'tattoo', 'temple', 'tiger', 'tower',
    'town', 'toy', 'train', 'tree', 'valley', 'vehicle', 'water', 'waterfall',
    'wedding', 'whales', 'window', 'zebra'
]

urls = {
    'images': 'http://127.0.0.1/nuswide.tar.gz',
    'image_list': 'http://127.0.0.1/ImageList.zip',
}


def read_image_label(root, file, subset):
    print('[dataset] read ' + file)
    data = dict()
    # read label
    label = []
    with open(file, 'r') as f:
        for line in f:
            if int(line) > 0:
                label.append(1)
            else:
                label.append(0)

    # read file path
    name = []
    with open(
            os.path.join(root, 'nuswide','ImageList', subset + 'Imagelist.txt'), 'r') as f:
        for line in f:
            name.append(line.split('\n')[0])
    # set data
    for i in range(len(label)):
        data[name[i]] = label[i]
    return data


def read_object_labels(root, subset):
    path_labels = os.path.join(root, 'nuswide', 'Groundtruth', 'TrainTestLabels')
    labeled_data = dict()
    num_classes = len(object_categories)
    for i in range(num_classes):
        file = os.path.join(
            path_labels,
            'Labels_' + object_categories[i] + '_' + subset + '.txt')
        data = read_image_label(root, file, subset)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(len(object_categories)):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)
    csvfile.close()


def read_object_labels_csv(file, header=True):
    unlabeled = 0
    remaining = 0
    total = 0
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                total += 1
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(
                    np.float32)
                labels = torch.from_numpy(labels)
                if labels.sum() <= 0:
                    unlabeled += 1
                else:
                    remaining += 1
                    item = (name, labels)
                    images.append(item)
            rownum += 1
    print("Total : " + str(total))
    print("Unlabeled : " + str(unlabeled))
    print("Remaining : " + str(remaining))
    return images


def download_nus_wide(root):
    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    # original images
    parts = urlparse(urls['images'])
    filename = os.path.basename(parts.path)
    images_file = os.path.join(root, filename)
    images_path = os.path.join(root, 'nuswide')

    # download images
    if not os.path.exists(images_file):
        print('Downloading: "{}" to {}\n'.format(urls['images'], images_file))
        util.download_url(urls['images'], images_file)

    # extract images
    if not os.path.exists(images_path):
        print('[dataset] Extracting tar file {file} to {path}'.format(
            file=images_file, path=images_path))
        fz = tarfile.open(images_file)
        fz.extractall(images_path)
        fz.close()
        print('[dataset] Done!')

    # image list
    parts = urlparse(urls['image_list'])
    filename = os.path.basename(parts.path)
    list_file = os.path.join(root, filename)
    list_path = os.path.join(root, 'ImageList')

    # download lists
    if not os.path.exists(list_file):
        print('Downloading: "{}" to {}\n'.format(urls['images'], list_file))
        util.download_url(urls['images'], list_file)

    # extract lists
    if not os.path.exists(list_path):
        print('[annotation] Extracting tar file {file} to {path}'.format(
            file=list_file, path=list_path))
        shutil.unpack_archive(list_file, list_path)
        print('[annotation] Done!')


class NUSWIDEClassification(data.Dataset):
    has_selected = []

    def __init__(self, root, subset, transform=None, inp_name=None):
        self.root = root
        self.path_images = os.path.join(root, 'nuswide', 'Flickr')
        self.annotation = os.path.join(root, 'nuswide', 'Groundtruth')
        self.set = subset
        self.transform = transform
        self.target_transform = None

        # define filename of csv file
        file_csv = os.path.join(self.annotation, subset + '_classification.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            # download dataset
            # download_nus_wide(self.root)
            # generate csv file
            labeled_data = read_object_labels(self.root, self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)
        #
        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        # for path, target = self.images[index]
        for index in range(len(self.images)):
            path, target = self.images[index]
            if  '0098_1153198149' in path:
                print(target)
            if sum(target) < 0:
                print(path)
            
        
        # if inp_name is not None:
        #     with open(inp_name, 'rb') as f:
        #         self.inp = pickle.load(f)
        #     self.inp_name = inp_name

        print('[dataset] NUS-WIDE classification set=%s number of classes=%d  number of images=%d' % (subset, len(self.classes), len(self.images)))
        '''
        [dataset] NUS-WIDE classification set=Train number of classes=81  number of images=125449
        [dataset] NUS-WIDE classification set=Test number of classes=81  number of images=83898
        '''
        

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(
            os.path.join(self.path_images, path).replace('\\',
                                                         '/')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

        # return (img, path, self.inp), (2 * target - 1)

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        print(self.__len__())
        return len(self.classes)

if __name__ == '__main__':
    root = '/media/data/maleilei/MLICdataset/'

    train_dataset = NUSWIDEClassification(root, subset='Train', transform=None, inp_name=None)
    # print(train_dataset[0])
    test_dataset = NUSWIDEClassification(root, subset='Test', transform=None, inp_name=None)
