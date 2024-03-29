import os
from PIL import Image
import numpy as np
# from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation

defult_cfg = {


    "inputs": "rgbd",

    "dataset": "irseg",
    "root": "/home/wangyike/PST900_RGBT_Dataset",
    "n_classes": 9,
    "id_unlabel": -1,
    "brightness": 0.5,
    "contrast": 0.5,
    "saturation": 0.5,
    "p": 0.5,
    "scales_range": "0.5 2.0",
    "crop_size": "480 640",
    "eval_scales": "0.5 0.75 1.0 1.25 1.5 1.75",
    "eval_flip": "true",

    "ims_per_gpu": 4,
    "num_workers": 1,

    "lr_start": 5e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_power": 0.9,
    "epochs": 400,

    "loss": "crossentropy",
    "class_weight": "enet"
}


class IRSeg(data.Dataset):

    def __init__(self, cfg=defult_cfg, root=defult_cfg['root'], mode='train', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = root
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])
        self.resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.4537, 44.2457, 31.6650, 46.4071, 30.1391])
            self.binary_class_weight = np.array([1.5121, 10.2388])  # 未修改
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        self.infos = os.listdir(os.path.join(cfg['root'], mode, 'rgb'))
        self.mode = mode

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        image = Image.open(os.path.join(self.root, self.mode, 'rgb', image_path))
        depth = Image.open(os.path.join(self.root, self.mode, 'thermal', image_path)).convert('RGB')
        label = Image.open(os.path.join(self.root, self.mode, 'labels', image_path))
        if self.mode == 'train':
            bound = Image.open(os.path.join(self.root, self.mode, 'bound', image_path))

            sample = {
                'image': image,
                'depth': depth,
                'label': label,
                'bound': bound,
            }
            sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        else:
            sample = {
                'image': image,
                'depth': depth,
                'label': label
            }

        if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)
        else:
            sample = self.resize(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()

        sample['label_path'] = image_path.strip().split('/')[-1]  # + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            [0, 0, 0],  # background
            [0, 0, 255],  # fire_extinguisher
            [0, 255, 0],  # backpack
            [255, 0, 0],  # drill
            [255, 255, 255],  # survivor/rescue_randy
        ]


if __name__ == '__main__':
    trainset = IRSeg(mode='train')
    print(trainset.__len__())
    trainset = IRSeg(mode='test')
    print(trainset.__len__())
