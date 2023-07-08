import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.vocdataset import VOC2007, VOC2012
from dataset.voc07 import Voc07Dataset
from dataset.voc12 import Voc12Dataset
from dataset.vg500dataset import VGDataset
from dataset.nus_wide import NusWideDataset
from dataset.nus_wide_csv import NUSWIDEClassification
from dataset.nus_wide_asl import get_datasets_from_csv
from utils.cutout import SLCutoutPIL
from utils.crop import MultiScaleCrop
from randaugment import RandAugment
import os.path as osp

def distributedsampler(args, train_dataset, val_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader, train_sampler

def get_datasets(args):
    if args.crop:
        train_data_transform_list = [transforms.Resize((args.img_size+64, args.img_size+64)),
                                                MultiScaleCrop(args.img_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
    else:
        train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                                RandAugment(),
                                                transforms.ToTensor()]

    test_data_transform_list =  [transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor()]
    if args.cutout and args.crop is not True:
        print("Using Cutout!!!")
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    
    if args.remove_norm is False:
        if args.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0],
                                            std=[1, 1, 1])
            print("mean=[0, 0, 0], std=[1, 1, 1]")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        train_data_transform_list.append(normalize)
        test_data_transform_list.append(normalize)
    else:
        print('remove normalize')

    train_data_transform = transforms.Compose(train_data_transform_list)
    # train_data_transform = transforms.Compose(test_data_transform_list)
    test_data_transform = transforms.Compose(test_data_transform_list)

    if args.dataname == 'coco14' or args.dataname == 'COCO2014':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        dataset_dir = osp.join(dataset_dir, 'COCO2014')
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'train_label_vectors_coco14.npy')
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'val_label_vectors_coco14.npy')
        )
    elif args.dataname == 'coco17' or args.dataname == 'COCO2017':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        dataset_dir = osp.join(dataset_dir, 'COCO2017')
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2017'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2017.json'),
            input_transform=train_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'train_label_vectors_coco17.npy')
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2017'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2017.json'),
            input_transform=test_data_transform,
            labels_path= osp.join(dataset_dir, 'label_npy', 'val_label_vectors_coco17.npy')
        )

    elif args.dataname == 'nus_asl' or args.dataname == 'nus_wide_asl':
        dataset_dir = args.dataset_dir
        nus_root = osp.join(dataset_dir, 'NUS_WIDE') 
        dataset_local_path = nus_root # osp.join(nus_root,'nuswide')
        metadata_local_path = nus_root
        json_path = osp.join(nus_root,'benchmark_81_v0.json')

        train_dataset, val_dataset, train_cls_ids, test_cls_ids = get_datasets_from_csv(
                        dataset_local_path = dataset_local_path,
                        metadata_local_path = metadata_local_path, 
                        train_transform = train_data_transform,
                        val_transform = test_data_transform, 
                        json_path = json_path)
        # len(train_dataset): 119103  len(val_dataset): 50720 total
        del train_cls_ids
        del test_cls_ids

    elif args.dataname == 'nus' or args.dataname == 'nus_wide':
        dataset_dir = args.dataset_dir
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

    elif args.dataname == 'nus_csv' or args.dataname == 'nus_wide_csv':
        dataset_dir = args.dataset_dir
        train_dataset = NUSWIDEClassification(root=dataset_dir, 
                                              subset='Train', 
                                              transform=train_data_transform, 
                                              inp_name=None)
        val_dataset = NUSWIDEClassification(root=dataset_dir, 
                                              subset='Test', 
                                              transform=test_data_transform, 
                                              inp_name=None)

        # len(train_dataset): 119103  len(val_dataset): 50720 total

    elif args.dataname == 'voc2007' or args.dataname == 'VOC2007':
        dataset_dir = osp.join(args.dataset_dir, 'VOC2007')

        # print(dataset_dir)
        # print('=='*30)
        # train_dataset = VOC2012(dataset_dir, phase='trainval', transform=train_data_transform)
        # val_dataset = VOC2012(dataset_dir, phase='test', transform=test_data_transform)

        dup=None
        train_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
                                    transform = train_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                    dup=None)

        val_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
                                    transform = test_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                    dup=None)

    elif args.dataname == 'voc2012' or args.dataname == 'VOC2012':
        dataset_dir = osp.join(args.dataset_dir, 'VOC2012')
        # print(dataset_dir)
        # print('=='*30)
        # train_dataset = VOC2012(dataset_dir, phase='train', transform=train_data_transform)
        # val_dataset = VOC2012(dataset_dir, phase='val', transform=test_data_transform)
        # test_dataset = VOC2012(dataset_dir, phase='test', transform=test_data_transform)

        dup=None
        # define dataset
        # train_dataset = Voc2012Classification(args.data, 'train', inp_name='data/voc2012/voc2012_glove_word2vec.pkl')
        # val_dataset = Voc2012Classification(args.data, 'val', inp_name='data/voc2012/voc2012_glove_word2vec.pkl')
        # test_dataset = Voc2012Classification(args.data, 'test', inp_name='data/voc2012/voc2012_glove_word2vec.pkl')
        train_dataset = Voc12Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'), 
                                    transform = train_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
                                    dup=None)

        # val_dataset = Voc12Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
        #                             anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/val.txt'), 
        #                             transform = test_data_transform, 
        #                             labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
        #                             dup=None)
        dataset_dir = '/media/mlldiskSSD/MLICdataset/VOC2007'
        val_dataset = Voc07Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
                                    transform = train_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
                                    dup=None)
        if args.evaluate:
            dataset_dir = '/media/mlldiskSSD/MLICdataset/VOC2012'
            test_dataset = Voc12Dataset(img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
                                    anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/test.txt'), 
                                    transform = test_data_transform, 
                                    labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
                                    dup=None)
            del train_dataset
            del val_dataset
            print("len(test_dataset):", len(test_dataset)) 
            return test_dataset
    
    elif args.dataname == 'vg' or args.dataname == 'vg500':
        dataset_dir = args.dataset_dir
        vg_root = osp.join(dataset_dir, 'VG')
        train_dir = osp.join(vg_root,'VG_100K')
        train_list = osp.join(vg_root,'train_list_500.txt')
        test_dir = osp.join(vg_root,'VG_100K')
        test_list = osp.join(vg_root,'test_list_500.txt')
        train_label = osp.join(vg_root,'vg_category_500_labels_index.json')
        test_label = osp.join(vg_root,'vg_category_500_labels_index.json')

        train_dataset = VGDataset(train_dir, train_list, train_data_transform, train_label)
        val_dataset = VGDataset(test_dir, test_list, test_data_transform, test_label)

    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
