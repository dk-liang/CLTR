from __future__ import division

import os
import warnings

from config import return_args, args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import get_root_logger, setup_seed
import nni
from nni.utils import merge_parameter
import time
import util.misc as utils
from utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard

if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)


def main(args):
    if args['dataset'] == 'jhu':
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'nwpu':
        test_file = './npydata/nwpu_val.npy'

    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)

    model = model.cuda()

    model = nn.DataParallel(model, device_ids=[int(data) for data in list(args['gpu_id']) if data!=','])
    path = './save_file/log_file/debug/'
    args['save_path'] = path
    if not os.path.exists(args['save_path']):
        os.makedirs(path)
    logger = get_root_logger(path + 'debug.log')
    writer = SummaryWriter(path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params / 1e6)
    logger.info("model params: = {:.3f}\t".format(num_params / 1e6))

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    if args['local_rank'] == 0:
        logger.info(args)

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    print('best result:', args['best_pred'])
    logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] == 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(args['best_pred'], args['start_epoch']))

    test_data = test_list
    if args['local_rank'] == 0:
        logger.info('start training!')

    eval_epoch = 0

    pred_mae, pred_mse, visi = validate(test_data, model, criterion, logger, args)

    writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
    writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)

    # save_result
    if args['save']:
        is_best = pred_mae < args['best_pred']
        args['best_pred'] = min(pred_mae, args['best_pred'])
        save_checkpoint({
            'arch': args['pre'],
            'state_dict': model.state_dict(),
            'best_prec1': args['best_pred'],
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, args['save_path'])

    if args['local_rank'] == 0:
        logger.info(
            'mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'.format(
                args['epochs'],
                pred_mae, pred_mse,
                args['best_pred']))


def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        if return_args.train_patch:
            fname.append(item[0])

            for i in range(0, len(item[1])):
                imgs.append(item[1][i])

            for i in range(0, len(item[2])):
                targets.append(item[2][i])
        else:
            fname.append(item[0])
            imgs.append(item[1])
            targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def validate(Pre_data, model, criterion, logger, args):
    if args['local_rank'] == 0:
        logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []

    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)

        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)

        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args['num_queries'], dim=1)
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args['threshold']] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count

        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)

        if i % 30 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))

    print('mae', mae, 'mse', mse)
    return mae, mse, visi


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
