""" Training function of Self-Ensembling Semi-Supervised 3D Object Detection

Author: Zhao Na
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import votenet
from pytorch_utils import BNMomentumScheduler
from tf_visualizer import Visualizer as TfVisualizer
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
import loss_helper_sess
import ramps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--labeled_sample_list', default='scannetv2_train_0.3.txt',
                                            help='Labeled sample list from a certain percentage of training [static]')
parser.add_argument('--detector_checkpoint', default='./log_scannet/votenet/checkpoint.tar')
parser.add_argument('--log_dir', default='./log_scannet/sess',
                                            help='Dump dir to save model checkpoint [example: ./log_sunrgbd/sess_0.1]')
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in Votenet input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in Votenet input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--num_target', type=int, default=128, help='Proposal number [default: 128]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='seed_fps',
                            help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')

parser.add_argument('--max_epoch', type=int, default=120, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', default='2,8', help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80',
                                            help='When to decay the learning rate (in epochs) [default: 80]')
parser.add_argument('--lr_decay_rates', default='0.1', help='Decay rates for lr decay [default: 0.1]')

parser.add_argument('--ema_decay',  type=float,  default=0.999, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
parser.add_argument('--consistency_weight', type=float, default=10.0, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
parser.add_argument('--consistency_rampup', type=int,  default=30,  metavar='EPOCHS', help='length of the consistency loss ramp-up')

parser.add_argument('--print_interval', type=int, default=20, help='batch inverval to print loss')
parser.add_argument('--eval_interval', type=int, default=9, help='epoch inverval to evaluate model')

FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print('\n************************** GLOBAL CONFIG BEG **************************')
batch_size_list = [int(x) for x in FLAGS.batch_size.split(',')]
BATCH_SIZE = batch_size_list[0] + batch_size_list[1]
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Init datasets and dataloaders
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset
    from sunrgbd_twostream_dataset import SunrgbdLabedledTwoStreamDataset, SunrgbdUnlabedledTwoStreamDataset
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    LABELED_DATASET = SunrgbdLabedledTwoStreamDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                        num_points=NUM_POINT,
                                                        augment=True,
                                                        use_color=FLAGS.use_color,
                                                        use_height=(not FLAGS.no_height),
                                                        use_v1 = (not FLAGS.use_sunrgbd_v2))
    UNLABELED_DATASET = SunrgbdUnlabedledTwoStreamDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                           num_points=NUM_POINT,
                                                           augment=True,
                                                           use_color=FLAGS.use_color,
                                                           use_height=(not FLAGS.no_height),
                                                           use_v1 = (not FLAGS.use_sunrgbd_v2))
    TEST_DATASET = SunrgbdDetectionVotesDataset('val',
                                                num_points=NUM_POINT, augment=False,
                                                use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
                                                use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset
    from scannet_twostream_dataset import ScannetLabedledTwoStreamDataset, ScannetUnlabedledTwoStreamDataset
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    LABELED_DATASET = ScannetLabedledTwoStreamDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                        num_points=NUM_POINT,
                                                        augment=True,
                                                        use_color=FLAGS.use_color,
                                                        use_height=(not FLAGS.no_height))
    UNLABELED_DATASET = ScannetUnlabedledTwoStreamDataset(labeled_sample_list=FLAGS.labeled_sample_list,
                                                           num_points=NUM_POINT,
                                                           augment=True,
                                                           use_color=FLAGS.use_color,
                                                           use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val',
                                            num_points=NUM_POINT, augment=False,
                                            use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
log_string('Dataset sizes: labeled-{0}; unlabeled-{1}; VALID-{2}'.format(len(LABELED_DATASET),
                                                            len(UNLABELED_DATASET), len(TEST_DATASET)))

LABELED_DATALOADER = DataLoader(LABELED_DATASET, batch_size=batch_size_list[0],
                              shuffle=True, num_workers=batch_size_list[0], worker_init_fn=my_worker_init_fn)
UNLABELED_DATALOADER = DataLoader(UNLABELED_DATASET, batch_size=batch_size_list[1],
                              shuffle=True, num_workers=batch_size_list[1]//2, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn)


def create_model(ema=False):
    model = votenet.VoteNet(num_class=DATASET_CONFIG.num_class,
                            num_heading_bin=DATASET_CONFIG.num_heading_bin,
                            num_size_cluster=DATASET_CONFIG.num_size_cluster,
                            mean_size_arr=DATASET_CONFIG.mean_size_arr,
                            num_proposal=FLAGS.num_target,
                            input_feature_dim=num_input_channel,
                            vote_factor=FLAGS.vote_factor,
                            sampling=FLAGS.cluster_sampling)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

# init　networks
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detector = create_model()
ema_detector =  create_model(ema=True)

detector.to(device)
ema_detector.to(device)

train_detector_criterion = loss_helper_sess.get_detection_loss
train_consistency_criterion = loss_helper_sess.get_consistency_loss
test_detector_criterion = votenet.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(detector.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
if FLAGS.detector_checkpoint is not None and os.path.isfile(FLAGS.detector_checkpoint):
    checkpoint = torch.load(FLAGS.detector_checkpoint)
    pretrained_dict = checkpoint['model_state_dict']
    detector.load_state_dict(pretrained_dict)
    ema_detector.load_state_dict(pretrained_dict)
    epoch = checkpoint['epoch']
    print("Loaded votenet checkpoint %s (epoch: %d)" % (FLAGS.detector_checkpoint, epoch))


# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(detector, bn_lambda=bn_lbmd, last_epoch=-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TFBoard Visualizers
TRAIN_VISUALIZER = TfVisualizer(LOG_DIR, 'train')
TEST_VISUALIZER = TfVisualizer(LOG_DIR, 'test')

# Used for Pseudo box generation and AP calculation
CONFIG_DICT = {'dataset_config': DATASET_CONFIG,
               'remove_empty_box': False, 'use_3d_nms': True,
               'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
               'per_class_proposal': True, 'conf_thresh': 0.05}

print('************************** GLOBAL CONFIG END **************************')
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return FLAGS.consistency_weight * ramps.sigmoid_rampup(epoch, FLAGS.consistency_rampup)


def train_one_epoch(global_step):
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    detector.train()  # set model to training mode
    ema_detector.train()
    consistency_weight = get_current_consistency_weight(EPOCH_CNT)
    log_string('Current consistency weight: %f' % consistency_weight)

    unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER)

    for batch_idx, batch_data_label in enumerate(LABELED_DATALOADER):
        try:
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)
        except StopIteration:
            unlabeled_dataloader_iterator = iter(UNLABELED_DATALOADER)
            batch_data_unlabeled = next(unlabeled_dataloader_iterator)

        for key in batch_data_unlabeled:
            batch_data_label[key] = torch.cat((batch_data_label[key], batch_data_unlabeled[key]), dim=0)

        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {'point_clouds': batch_data_label['point_clouds']}
        ema_inputs = {'point_clouds': batch_data_label['ema_point_clouds']}

        optimizer.zero_grad()

        end_points = detector(inputs)
        ema_end_points = ema_detector(ema_inputs)

        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        detection_loss, end_points = train_detector_criterion(end_points, DATASET_CONFIG)
        consistency_loss, end_points = train_consistency_criterion(end_points, ema_end_points, DATASET_CONFIG)

        loss = detection_loss + consistency_loss * consistency_weight
        end_points['loss'] = loss
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(detector, ema_detector, FLAGS.ema_decay, global_step)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = FLAGS.print_interval
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            TRAIN_VISUALIZER.log_scalars({key: stat_dict[key] / batch_interval for key in stat_dict},
                                         (EPOCH_CNT * len(LABELED_DATALOADER) + batch_idx) * BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

    return global_step


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
                                 class2type_map=DATASET_CONFIG.class2type)
    detector.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = detector(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = test_detector_criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # Log statistics
    TEST_VISUALIZER.log_scalars({key: stat_dict[key] / float(batch_idx + 1) for key in stat_dict},
                                (EPOCH_CNT + 1) * len(LABELED_DATALOADER) * BATCH_SIZE)
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f' % (key, metrics_dict[key]))

    mean_loss = stat_dict['detection_loss'] / float(batch_idx + 1)
    return mean_loss


def train():
    global EPOCH_CNT
    global_step = 0
    loss = 0
    for epoch in range(0, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('\n**** EPOCH %03d, STEP %d ****' % (epoch, global_step))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        global_step = train_one_epoch(global_step)

        if EPOCH_CNT > 0 and  EPOCH_CNT % FLAGS.eval_interval == 0:
            loss = evaluate_one_epoch()
        # save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = detector.module.state_dict()
            save_dict['ema_model_state_dict'] = ema_detector.module.state_dict()
        except:
            save_dict['model_state_dict'] = detector.state_dict()
            save_dict['ema_model_state_dict'] = ema_detector.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))



if __name__ == '__main__':
    train()
