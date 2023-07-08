import os, sys
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class NusWideDataset(Dataset):
    def __init__(self, 
        img_dir,
        anno_path, 
        labels_path,
        transform=None,
        rm_no_label_data=True
        ) -> None:
        """[summary]
        Args:
            img_dir ([type]): dir of imgs
            anno_path ([type]): list of used imgs
            labels_path ([type]): labels of used imgs
            transform ([type], optional): [description]. Defaults to None.
            """
        super().__init__()

        self.img_dir = img_dir
        self.anno_path = anno_path
        self.labels_path = labels_path
        self.transform = transform
        self.rm_no_label_data = rm_no_label_data

        self.itemlist = self.preprocess() # [(imgpath, label),...]
        self.num_labels = 81


        # self.labels = []
        # for idx, (imgpath, labels) in enumerate(self.itemlist):
        #     self.labels.append(labels)

        # self.temp = torch.from_numpy(np.array(self.labels))

    def preprocess(self):
        imgnamelist = [line.strip().replace('\\', '/') for line in open(self.anno_path, 'r')]
        labellist = [line.strip() for line in open(self.labels_path, 'r')]
        assert len(imgnamelist) == len(labellist)

        res = []
        for idx, (imgname, labelline) in enumerate(zip(imgnamelist, labellist)):
            imgpath = osp.join(self.img_dir, imgname)
            labels = [int(i) for i in labelline.split(' ')]
            labels = np.array(labels).astype(np.float32)

            if sum(labels) == 0:
                continue
            
            res.append((imgpath, labels))

        return res
    
    def __len__(self) -> int:
        return len(self.itemlist)

    def __getitem__(self, index: int):
        imgpath, labels = self.itemlist[index]

        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, labels


# /media/data/maleilei/MLICdataset/nus_wide
if __name__ == '__main__':
    import os.path as osp
    dataset_dir = '/media/data/maleilei/MLICdataset'
    train_data_transform = None
    test_data_transform = None
    nus_root = osp.join(dataset_dir, 'nuswide')

    train_dir = osp.join(nus_root,'Flickr')
    train_list = osp.join(nus_root, 'ImageList', 'TrainImagelist.txt')
    train_label = osp.join(nus_root, 'Groundtruth','Labels_Train.txt')

        #  Test_label.txt  Test_split.txt  Train_label.txt  Train_split.txt
    test_dir = osp.join(nus_root,'Flickr')
    test_list = osp.join(nus_root, 'ImageList', 'TestImagelist.txt')
    test_label = osp.join(nus_root, 'Groundtruth','Labels_Test.txt')

    train_dataset = NusWideDataset(
            img_dir=train_dir,
            anno_path=train_list,
            labels_path=train_label,
            transform = train_data_transform)

    val_dataset = NusWideDataset(
            img_dir=test_dir,
            anno_path=test_list,
            labels_path=test_label,
            transform = test_data_transform)



    # print(train_dataset.temp.sum() / len(train_dataset))

    # print( (train_dataset.temp.sum() + val_dataset.temp.sum()) / (len(train_dataset)+ len(val_dataset))  )

    print('[dataset] NUS-WIDE classification set=%s number of classes=%d  number of images=%d' % ('Train', train_dataset.num_labels, len(train_dataset)))
    print('[dataset] NUS-WIDE classification set=%s number of classes=%d  number of images=%d' % ('Test', val_dataset.num_labels, len(val_dataset)))

    # print("len(ds):", len(ds)) 
    # print("len(ds_test):", len(ds_test))
    # labels_path = '/media/data/maleilei/MLICdataset/nuswide/Groundtruth/Labels_Train.txt' # 161789
    # labels_path = '/media/data/maleilei/MLICdataset/nuswide/slsplit/Train_label.txt' # 150000

    


    # labellist = [line.strip() for line in open(labels_path, 'r')]
    # print(len(labellist))

    # anno_path = '/media/data/maleilei/MLICdataset/nuswide/oldtxt/train-file-list.txt'
    # imgnamelist = [line.strip().replace('\\', '/') for line in open(anno_path, 'r')]
    # print(len(imgnamelist))


    # ds = NusWideDataset(
    #         img_dir='/data/shilong/data/nus_wide/nuswide/Flickr',
    #         anno_path='/data/shilong/data/nus_wide/nuswide/ImageList/TrainImagelist.txt',
    #         labels_path='/data')