import torch
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from xml.dom.minidom import parse 
import xml.dom.minidom
import os
import os.path as osp
category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class Voc12Dataset(data.Dataset):
    def __init__(self, 
        img_dir='/data/shilong/data/pascal_voc/2012/VOCdevkit/VOC2012/JPEGImages', 
        anno_path='/data/shilong/data/pascal_voc/2012/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', 
        transform=None, 
        labels_path='/data/shilong/data/pascal_voc/2012/VOCdevkit/VOC2012/Annotations',
        dup=None,
    ):
        self.dup = dup
        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir
        self.labels = []
        self.return_name = False
        self.num_labels = len(category_info)

        if 'test' in os.path.split(anno_path)[-1]:
            # no ground truth of test data of voc12, just a placeholder
            self.labels = np.ones((len(self.img_names),20))
            self.return_name = True
        else:
            no_anno_cnt = 0
            res_name_list = []
            for name in self.img_names:
                label_file = os.path.join(labels_path,name[:-1]+'.xml')
                if not os.path.exists(label_file):
                    no_anno_cnt += 1
                    if no_anno_cnt < 10:
                        print("cannot find: %s" % label_file)
                    continue
                res_name_list.append(name)
                label_vector = np.zeros(20)
                DOMTree = xml.dom.minidom.parse(label_file)
                root = DOMTree.documentElement
                objects = root.getElementsByTagName('object')
                for obj in objects:
                    if (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                        continue
                    tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                    label_vector[int(category_info[tag])] = 1.0
                self.labels.append(label_vector)
            self.labels = np.array(self.labels).astype(np.float32)
            self.img_names = res_name_list
            if no_anno_cnt > 0:
                print("total no anno file count:", no_anno_cnt)

        self.transform = transform

        self.temp = torch.from_numpy(self.labels)


    def __getitem__(self, index):
        if self.dup:
            index = index % self.dup
        name = self.img_names[index][:-1]+'.jpg'
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
          
        if self.transform:
            input = self.transform(input)
        if self.return_name:
            name = name[:-4]
            return input, self.labels[index], name
        return input, self.labels[index]

    def __len__(self):
        if not self.dup:
            return len(self.img_names)
        else:
            return len(self.img_names) * self.dup

if __name__ == "__main__":
    import os.path as osp   
    dataset_dir = '/media/data/maleilei/MLICdataset'
    dataset_dir = osp.join(dataset_dir, 'VOC2012')
    train_data_transform = None
    test_data_transform = None
    train_dataset = Voc12Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'), 
                                    transform = train_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
                                    dup=None)

    val_dataset = Voc12Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/test.txt'), 
                                    transform = test_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
                                    dup=None)
    
    print(train_dataset.temp.sum() / len(train_dataset))

    print('[dataset] PASCAL VOC2012 classification set=%s number of classes=%d  number of images=%d' % ('Train', train_dataset.num_labels, len(train_dataset)))
    print('[dataset] PASCAL VOC2012 classification set=%s number of classes=%d  number of images=%d' % ('Test', val_dataset.num_labels, len(val_dataset)))
    print(print(val_dataset[20]))
