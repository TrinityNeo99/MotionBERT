import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data_coco as flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
# from lib.data.dataset_motion_3d_binocular_depth import MotionDataset3D  # dataloader depth
from lib.data.dataset_motion_3d_binocular import MotionDataset3D  # dataloader
from lib.data.dataset_motion_3d import MotionDataset3D as MotionDatasetH36m  # dataloader
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
# from lib.data.datareader_binocular_depth import DataReaderBinocular  # datareader depth
from lib.data.datareader_binocular import DataReaderBinocular  # datareader
from lib.data.datareader_h36m import DataReaderH36M  # datareader
from lib.model.loss import *
from train import evaluate as evaluateH36m
import wandb
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH',
                        help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME',
                        help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss
    }, chk_path)


def get_sliced_data_sub(data, ranges: list):
    return np.stack([data[r] for r in ranges], axis=0)


def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()
    with torch.no_grad():
        for batch_input, batch_input_right, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available() and torch.cuda.get_device_name(
                    args.device_ids[0]) != "NVIDIA GeForce RTX 3090":
                batch_input = batch_input.cuda()
                batch_input_right = batch_input_right.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]

            if args.test_task == "binocular":
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    batch_input_flip_right = flip_data(batch_input_right)
                    input_merge = torch.cat((batch_input, batch_input_right), dim=1)
                    predicted_3d_pos_1 = model_pos(input_merge, "binocular")  # (N, T, 17, 3)

                    input_flip = torch.cat((batch_input_flip, batch_input_flip_right), dim=1)
                    predicted_3d_pos_flip = model_pos(input_flip, "binocular")  # (N, T, 17, 3)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
                else:
                    input_merge = torch.cat((batch_input, batch_input_right), dim=1)
                    predicted_3d_pos = model_pos(input_merge, "binocular")  # (N, T, 17, 3)

            elif args.test_task == "binocular_separate":
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = model_pos(batch_input, "monocular")  # (N, T, 17, 3)
                    predicted_3d_pos_flip = model_pos(batch_input_flip, "monocular")  # (N, T, 17, 3)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
                else:
                    predicted_3d_pos = model_pos(batch_input, "monocular")  # (N, T, 17, 3)

            elif args.test_task == "binocular_spatial":
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    batch_input_flip_right = flip_data(batch_input_right)
                    input_merge = torch.cat((batch_input, batch_input_right), dim=2)
                    predicted_3d_pos_1 = model_pos(input_merge, "binocular_spatial")  # (N, T, 17, 3)

                    input_flip = torch.cat((batch_input_flip, batch_input_flip_right), dim=2)
                    predicted_3d_pos_flip = model_pos(input_flip, "binocular_spatial")  # (N, T, 17, 3)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
                else:
                    input_merge = torch.cat((batch_input, batch_input_right), dim=2)
                    predicted_3d_pos = model_pos(input_merge, "binocular_spatial")  # (N, T, 17, 3)
            else:
                raise Exception("Undefined task type")

            if args.rootrel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                batch_gt[:, 0, 0, 2] = 0

            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    # results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    # factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joint3d_image'])
    # sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    # action_clips = actions[split_id_test]
    action_clips = get_sliced_data_sub(actions, split_id_test)
    # factor_clips = factors[split_id_test]
    # source_clips = sources[split_id_test]
    # frame_clips = frames[split_id_test]
    frame_clips = get_sliced_data_sub(frames, split_id_test)
    # gt_clips = gts[split_id_test]
    gt_clips = get_sliced_data_sub(gts, split_id_test)
    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []

    for idx in range(len(action_clips)):
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        # factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        # pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
    final_result = []
    final_result_procrustes = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1 * 1000, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2 * 1000, 'mm')
    print('----------')
    return e1, e2, results_all


def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    device = f"cuda:{args.device_ids[0]}"
    model_pos.train()
    for idx, (batch_input, batch_input_right, batch_gt) in tqdm(enumerate(train_loader)):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.to(device)
            batch_input_right = batch_input_right.to(device)
            batch_gt = batch_gt.to(device)
        batch_input, batch_input_right, batch_gt = batch_preprocess(batch_input, batch_input_right, batch_gt, args,
                                                                    has_3d,
                                                                    has_gt)
        # Predict 3D poses
        for task in args.tasks:
            # TODO 可以加一个判断条件，根据数据的类型判断是 binocular 还是 monocular
            if task == "binocular":
                input_merge = torch.cat((batch_input, batch_input_right), dim=1)
                predicted_3d_pos = model_pos(input_merge, task)  # (N, T, 17, 3)
            elif task == "binocular_separate":
                predicted_3d_pos = model_pos(batch_input, "monocular")  # (N, T, 17, 3)
                predicted_3d_pos_ = model_pos(batch_input_right, "monocular")  # (N, T, 17, 3)
            elif task == "binocular_spatial":
                input_merge = torch.cat((batch_input, batch_input_right), dim=2)  # concat along spatial
                predicted_3d_pos = model_pos(input_merge, task)  # (N, T, 34, 3)
            elif task == "monocular":
                predicted_3d_pos = model_pos(batch_input, task)
            elif task == "left2right":
                predict_2d_right = model_pos(batch_input, task)
            elif task == "right2left":
                predict_2d_left = model_pos(batch_input_right, task)
            elif task == "self2self":
                drop_out = nn.Dropout(p=0.2)
                input = drop_out(batch_input)
                predict_2d_self = model_pos(input, task)

        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)

            if "left2right" in args.tasks:
                loss_r2l = loss_mpjpe(predict_2d_left, batch_input)
                losses['r2l'].update(loss_r2l.item(), batch_size)
            else:
                loss_r2l = 0

            if "right2left" in args.tasks:
                loss_l2r = loss_mpjpe(predict_2d_right, batch_input_right)
                losses['l2r'].update(loss_l2r.item(), batch_size)
            else:
                loss_l2r = 0

            if "self2self" in args.tasks:
                loss_s2s = loss_mpjpe(batch_input, predict_2d_self)
                losses['s2s'].update(loss_s2s.item(), batch_size)
            else:
                loss_s2s = 0

            loss_total = loss_3d_pos + \
                         args.lambda_scale * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv * loss_lv + \
                         args.lambda_lg * loss_lg + \
                         args.lambda_a * loss_a + \
                         args.lambda_av * loss_av + \
                         args.lambda_r2l * loss_r2l + \
                         args.lambda_l2r * loss_l2r + args.lambda_s2s * loss_s2s

            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()


def batch_preprocess(batch_input, batch_input_right, batch_gt, args, has_3d, has_gt):
    with torch.no_grad():
        if args.no_conf:
            batch_input = batch_input[:, :, :, :2]
            if batch_input_right is not None:
                batch_input_right = batch_input_right[:, :, :, :2]
        if not has_3d:
            conf = copy.deepcopy(batch_input[:, :, :, 2:])  # For 2D data, weight/confidence is at the last channel
        if args.rootrel:
            batch_gt = batch_gt - batch_gt[:, :, 0:1, :]
        else:
            batch_gt[:, :, :, 2] = batch_gt[:, :, :, 2] - batch_gt[:, 0:1, 0:1,
                                                          2]  # Place the depth of first frame root to 0.
        if args.mask or args.noise:
            batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
            if batch_input_right is not None:
                batch_input_right = args.aug.augment2D(batch_input_right, noise=(args.noise and has_gt), mask=args.mask)

    return batch_input, batch_input_right, batch_gt


def train_monocular(args, model_pos, losses, optimizer, has_3d, has_gt, batch_input, batch_gt):
    device = f"cuda:{args.device_ids[0]}"
    if torch.cuda.is_available():
        batch_input = batch_input.to(device)
        batch_gt = batch_gt.to(device)
    batch_input, _, batch_gt = batch_preprocess(batch_input, None, batch_gt, args, has_3d, has_gt)
    predicted_3d_pos = model_pos(batch_input, "monocular")
    optimizer.zero_grad()
    batch_size = len(batch_input)
    if has_3d:
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)

        loss_total = loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     args.lambda_a * loss_a + \
                     args.lambda_av * loss_av
        losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
    else:
        loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
        loss_total = loss_2d_proj
        losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
    return loss_total


def train_binocular(args, model_pos, losses, optimizer, has_3d, has_gt, batch_input, batch_input_right, batch_gt):
    device = f"cuda:{args.device_ids[0]}"
    if torch.cuda.is_available():
        batch_input = batch_input.to(device)
        batch_input_right = batch_input_right.to(device)
        batch_gt = batch_gt.to(device)
    batch_input, batch_input_right, batch_gt = batch_preprocess(batch_input, batch_input_right, batch_gt, args, has_3d,
                                                                has_gt)
    for task in args.tasks:
        if task == "binocular":
            input_merge = torch.cat((batch_input, batch_input_right), dim=1)
            predicted_3d_pos = model_pos(input_merge, task)  # (N, T, 17, 3)
        elif task == "binocular_separate":
            predicted_3d_pos = model_pos(batch_input, "monocular")  # (N, T, 17, 3)
            predicted_3d_pos_ = model_pos(batch_input_right, "monocular")  # (N, T, 17, 3)
        elif task == "binocular_spatial":
            input_merge = torch.cat((batch_input, batch_input_right), dim=2)  # concat along spatial
            predicted_3d_pos = model_pos(input_merge, task)  # (N, T, 34, 3)
        elif task == "left2right":
            predict_2d_right = model_pos(batch_input, task)
        elif task == "right2left":
            predict_2d_left = model_pos(batch_input_right, task)
        elif task == "self2self":
            drop_out = nn.Dropout(p=0.2)
            input = drop_out(batch_input)
            predict_2d_self = model_pos(input, task)
        elif task == "monocular":
            pass  # do nothing
        else:
            raise Exception("No Implementation")
    optimizer.zero_grad()
    batch_size = len(batch_input)
    if has_3d:
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)

        if "left2right" in args.tasks:
            loss_r2l = loss_mpjpe(predict_2d_left, batch_input)
            losses['r2l'].update(loss_r2l.item(), batch_size)
        else:
            loss_r2l = 0

        if "right2left" in args.tasks:
            loss_l2r = loss_mpjpe(predict_2d_right, batch_input_right)
            losses['l2r'].update(loss_l2r.item(), batch_size)
        else:
            loss_l2r = 0

        if "self2self" in args.tasks:
            loss_s2s = loss_mpjpe(batch_input, predict_2d_self)
            losses['s2s'].update(loss_s2s.item(), batch_size)
        else:
            loss_s2s = 0

        loss_total = loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     args.lambda_a * loss_a + \
                     args.lambda_av * loss_av + \
                     args.lambda_r2l * loss_r2l + \
                     args.lambda_l2r * loss_l2r + args.lambda_s2s * loss_s2s

        losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
    else:
        loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
        loss_total = loss_2d_proj
        losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
    return loss_total


def train_epoch_mix_single_dataset_per_batch(args, model_pos, monocular_dataloader, binocular_dataloader, losses,
                                             optimizer, has_3d, has_gt):
    model_pos.train()
    monocular_step = len(monocular_dataloader)
    ratio = args.mo_bi_ratio
    total_step = monocular_step + monocular_step // ratio
    monocular_dataloader_iter = iter(monocular_dataloader)
    binocular_dataloader_iter = iter(binocular_dataloader)
    with tqdm(total=total_step) as pbar:
        pbar.set_description("training...")
        for i in range(total_step):
            if i % (ratio + 1) < ratio:
                # 使用第一个数据集
                try:
                    batch_input, batch_gt = next(monocular_dataloader_iter)
                except StopIteration:
                    # 重新开始第一个数据集迭代
                    monocular_dataloader_iter = iter(monocular_dataloader)
                    batch_input, batch_gt = next(monocular_dataloader_iter)
                loss_total = train_monocular(args, model_pos, losses, optimizer, has_3d, has_gt, batch_input, batch_gt)
            else:
                # 使用第二个数据集
                try:
                    batch_input, batch_input_right, batch_gt = next(binocular_dataloader_iter)
                except StopIteration:
                    # 重新开始第二个数据集迭代
                    binocular_dataloader_iter = iter(binocular_dataloader)
                    batch_input, batch_input_right, batch_gt = next(binocular_dataloader_iter)
                loss_total = train_binocular(args, model_pos, losses, optimizer, has_3d, has_gt, batch_input,
                                             batch_input_right, batch_gt)
            pbar.update(1)
            loss_total.backward()
            optimizer.step()


def train_with_config(args, opts):
    print(args)
    print("*****************Training Start...")
    for d in args.device_ids:
        print(torch.cuda.get_device_name(d))
    try:
        if os.path.exists(opts.checkpoint) and not opts.evaluate:
            decision = input("There already exist a latest_epoch, delete? [y/n]")
            if decision == "y":
                shutil.rmtree(opts.checkpoint)
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    print('Loading dataset...')
    trainloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    testloader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True
    }

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader_3d = DataLoader(test_dataset, **testloader_params)

    if "monocular" in args.tasks:
        # human 3.6 m
        temp_args = copy.deepcopy(args)
        temp_args.data_root = args.data_root_h36m
        temp_args.subset_list = args.subset_list_h36m
        temp_args.dt_file = args.dt_file_h36m
        train_dataset_h36m = MotionDatasetH36m(temp_args, temp_args.subset_list, "train")
        test_dataset_h36m = MotionDatasetH36m(temp_args, temp_args.subset_list, "test")
        train_loader_3d_h36m = DataLoader(train_dataset_h36m, **trainloader_params)
        test_loader_3d_h36m = DataLoader(test_dataset_h36m, **testloader_params)

        datareader_h36m = DataReaderH36M(n_frames=temp_args.clip_len, sample_stride=temp_args.sample_stride,
                                         data_stride_train=temp_args.data_stride, data_stride_test=temp_args.clip_len,
                                         dt_root=args.data_root_h36m,
                                         dt_file=temp_args.dt_file)

    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)

    datareader = DataReaderBinocular(n_frames=args.clip_len, sample_stride=args.sample_stride,
                                     data_stride_train=args.data_stride, data_stride_test=args.clip_len,
                                     dt_root='../dataset', dt_file=args.dt_file)

    min_loss = 100000
    min_loss_e2 = 100000
    min_loss_h36m = 100000
    min_loss_e2_h36m = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone, args.device_ids)
        model_backbone = model_backbone.cuda(device=args.device_ids[0])

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        print(chk_filename)
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone

    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr,
                                weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d),
                                                                   len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
            if "monocular" in args.tasks:
                print('INFO: Training on {}(3D) batches H36M'.format(len(train_loader_3d_h36m)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']

        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)

        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            losses['r2l'] = AverageMeter()
            losses['l2r'] = AverageMeter()
            losses['s2s'] = AverageMeter()
            N = 0

            # Curriculum Learning
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
                train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            if not args.train_mix:
                train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True)
            elif args.train_mix and "monocular" in args.tasks:
                train_epoch_mix_single_dataset_per_batch(args, model_pos, train_loader_3d_h36m, train_loader_3d, losses,
                                                         optimizer, has_3d=True, has_gt=True)
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg))
            else:
                e1, e2, results_all = evaluate(args, model_pos, test_loader_3d, datareader)
                print('Sports: [%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))

                wandb.log({"Error P1": e1 * 1000, "epoch": epoch + 1})
                wandb.log({"Error P2": e2 * 1000, "epoch": epoch + 1})
                wandb.log({"loss_3d_pos": losses['3d_pos'].avg, "epoch": epoch + 1})
                wandb.log({"loss_2d_project": losses['2d_proj'].avg, "epoch": epoch + 1})
                wandb.log({"loss_3d_scale": losses['3d_scale'].avg, "epoch": epoch + 1})
                wandb.log({"loss_3d_velocity": losses['3d_velocity'].avg, "epoch": epoch + 1})
                wandb.log({"loss_lv": losses['lv'].avg, "epoch": epoch + 1})
                wandb.log({"loss_lg": losses['lg'].avg, "epoch": epoch + 1})
                wandb.log({"loss_angle": losses['angle'].avg, "epoch": epoch + 1})
                wandb.log({"loss_angle_velocity": losses['angle_velocity'].avg, "epoch": epoch + 1})
                wandb.log({"loss_r2l": losses['r2l'].avg, "epoch": epoch + 1})
                wandb.log({"loss_l2r": losses['l2r'].avg, "epoch": epoch + 1})
                wandb.log({"loss_s2s": losses['s2s'].avg, "epoch": epoch + 1})
                wandb.log({"loss_total": losses['total'].avg, "epoch": epoch + 1})
                wandb.log({"lr": lr, "epoch": epoch + 1})

                if "monocular" in args.tasks:
                    e1_h36m, e2_h36m, results_all = evaluateH36m(args, model_pos, test_loader_3d_h36m, datareader_h36m)
                    wandb.log({"H36m Error P1": e1_h36m, "epoch": epoch + 1})
                    wandb.log({"H36m Error P2": e2_h36m, "epoch": epoch + 1})
                    print('Human3.6N: [%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses['3d_pos'].avg,
                        e1_h36m, e2_h36m))

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
            if e2 < min_loss_e2:
                min_loss_e2 = e2
            print(
                f"Sports Binocular ==> The best results (minimal error) P1: {min_loss * 1000} mm, P2: {min_loss_e2 * 1000} mm")
            wandb.log({"Best Error P1": min_loss * 1000, "epoch": epoch + 1})
            wandb.log({"Best Error P2": min_loss_e2 * 1000, "epoch": epoch + 1})

            if "monocular" in args.tasks:
                # H3.6M
                if e1_h36m < min_loss_h36m:
                    min_loss_h36m = e1_h36m
                if e2_h36m < min_loss_e2_h36m:
                    min_loss_e2_h36m = e2_h36m
                print(
                    f"Human 3.6M ==> The best results (minimal error) P1: {min_loss_h36m} mm, P2: {min_loss_e2_h36m} mm")
                wandb.log({"H36m Best Error P1": min_loss_h36m, "epoch": epoch + 1})
                wandb.log({"H36m Best Error P2": min_loss_e2_h36m, "epoch": epoch + 1})

    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader_3d, datareader)
        print(f"Sports Binocular ==> The best results (minimal error) P1: {e1 * 1000} mm, P2: {e2 * 1000} mm")
        if "monocular" in args.tasks:
            e1_h36m, e2_h36m, results_all = evaluateH36m(args, model_pos, test_loader_3d_h36m, datareader_h36m)
            print(f"Human 3.6M ==> The best results (minimal error) P1: {e1_h36m} mm, P2: {e2_h36m} mm")


def wandb_init(args):
    wandb.login(key="610ea58ece04cbfb08fe53c2d852fccf1833d910", force=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="pose_estimation",
        # name="ASE_GCN_baseline",
        name=args.model_saved_name,
        # track hyperparameters and run metadata
        config=args
    )


if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    args.model_saved_name = os.path.basename(opts.checkpoint)
    wandb_init(args)
    train_with_config(args, opts)
