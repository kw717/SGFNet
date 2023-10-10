import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import torch.nn.functional as F
from tqdm import tqdm
from toolbox import get_dataset, load_ckpt  # loss
from toolbox.optim.Ranger import Ranger
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import save_ckpt
from toolbox import setup_seed
from toolbox.datasets.irseg import IRSeg

from toolbox.losses import lovasz_softmax


from SGFNet import SGFNet

import os
import warnings
warnings.filterwarnings('ignore')
########################在这设置超参数#########################
#setup_seed(99)
batch_size = 2
lr_start = 5e-5
weight_decay = 5e-4
epoch = 250
gpu = '0'
saveprefix = ''
loadmodel = ''
############################################################
class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        self.class_weight = class_weight
        # self.LovaszSoftmax = lovasz_softmax()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)

    def forward(self, inputs, targets):
        semantic_gt, binary_gt, boundary_gt = targets
        semantic_out, edge_out, sal_out = inputs

        loss1 = self.semantic_loss(semantic_out, semantic_gt)
        loss2 = lovasz_softmax(F.softmax(semantic_out, dim=1), semantic_gt, ignore=255)
        #loss3 = self.semantic_loss(semantic_out_2, semantic_gt)
        loss4 = self.semantic_loss(sal_out, semantic_gt)
        loss5 = self.boundary_loss(edge_out, boundary_gt)

        loss = loss1 + loss2 + 0.5 * loss4 + 2 * loss5  # ortgin 1 2 2

        return loss


def run():

    model = SGFNet(9).cuda()
    if gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device_ids = range(torch.cuda.device_count())
        device = torch.device('cuda')

        #model = torch.nn.DataParallel(model).cuda()
    else:
      device = torch.device('cuda:0')
      model.to(device)

    if loadmodel!='' :
        load_ckpt(model=model,perfix=loadmodel)



    trainset = IRSeg( mode='train')
    testset = IRSeg( mode='test')

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=lr_start, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / epoch) ** 0.9)


    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)


    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(9, ignore_index=-1)
    best_test = 0


    if gpu!='':
        model = torch.nn.DataParallel(model).cuda()


    for ep in range(epoch):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            label = sample['label'].to(device)
            bound = sample['bound'].to(device)
            #edge = sample['edge'].to(device)
            binary_label = sample['binary_label'].to(device)
            targets = [label, binary_label, bound]
            predict = model(image, depth)

            loss = train_criterion(predict, targets)
            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        if ep < 2 or ep % 10 == 0 or ep > 60:
          with torch.no_grad():
              model.eval()
              running_metrics_test.reset()
              test_loss_meter.reset()
              for i, sample in enumerate(test_loader):

                  image = sample['image'].to(device)
                  depth = sample['depth'].to(device)
                  label = sample['label'].to(device)
                  #edge = sample['edge'].to(device)
                  predict = model(image, depth)[0]

                  loss = criterion(predict, label)
                  test_loss_meter.update(loss.item())

                  predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                  label = label.cpu().numpy()
                  running_metrics_test.update(label, predict)

          train_loss = train_loss_meter.avg
          test_loss = test_loss_meter.avg

          test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
          test_miou = running_metrics_test.get_scores()[0]["mIou: "]
          #test_fwiou = running_metrics_test.get_scores()[0]["fwiou:"]
          test_avg = (test_macc + test_miou) / 2
          with open("info.txt", "a") as f:
             f.write(
              f'save{saveprefix}Iter | [{ep + 1:3d}/{epoch}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}\n')
          print(f'save{saveprefix}Iter | [{ep + 1:3d}/{epoch}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f},bset={best_test:.3f}')
          if test_miou > best_test:
              best_test = test_miou
              save_ckpt('', model,prefix=saveprefix)


if __name__ == '__main__':

    #torch.backends.cudnn.enabled = False
    run()



