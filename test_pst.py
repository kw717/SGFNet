import os
import time

import numpy
import numpy as np
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_model, compute_speed
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt, save_ckpt

from toolbox.datasets.pst900 import IRSeg
from SGFNet import SGFNet


def evaluate(args, logdir, save_predict=False, options=['val', 'test', 'test_day', 'test_night'], prefix=''):
    device = torch.device('cuda')

    loaders = []
    for opt in options:
        dataset = IRSeg(mode=opt)

        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)))
        cmap = dataset.cmap

    model = SGFNet(5).to(device)

    model.load_state_dict(torch.load(args.pth, map_location='cuda:0'))

    running_metrics_val = runningScore(9, ignore_index=-1)
    time_meter = averageMeter()
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)
    softmax = torch.nn.Softmax(dim=1)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#' * 50 + '    ' + name + prefix + '    ' + '#' * 50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

                time_start = time.time()

                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)

                predict = model(image, depth)[0]

                predict = softmax(predict)
                predict = predict.max(1)[1].cpu().numpy()

                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

                time_meter.update(time.time() - time_start, n=image.size(0))

                if save_predict:

                    for x in range(1):
                        p = predict
                        p = p[x, :, :]
                        p = class_to_RGB(p, N=len(cmap), cmap=cmap)
                        p = Image.fromarray(p)

                        p.save(os.path.join(save_path, sample['label_path'][x]))


        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v:.4f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v:.4f}')
        print('acc for each class .....')
        for k, v in metrics[2].items():
            print(k, f'{v:.4f}')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", default="", type=str,
                        help="run logdir")
    parser.add_argument("--data-root", default="", type=str,
                        help="path to dataset root")
    parser.add_argument("--pth", default="", type=str,
                        help="path to model weight")
    parser.add_argument("-s", type=bool, default=False,
                        help="save predict or not")
    parser.add_argument("-option", type=str, default="test",
                        help="choose test set. options=['test']")
    args = parser.parse_args()

    evaluate(args, args.logdir, options=[args.option], prefix='', save_predict=args.s)
