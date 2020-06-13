""" Loss Function for Self-Ensembling Semi-Supervised 3D Object Detection

Author: Zhao Na, 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

def compute_vote_loss(end_points, supervised_inds):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    batch_size_supervised = supervised_inds.shape[0]

    # Load ground truth votes and assign them to seed points
    seed_xyz = end_points['seed_xyz'][supervised_inds, ...] # B_l, num_seed, 3
    num_seed = seed_xyz.shape[1]
    vote_xyz = end_points['vote_xyz'][supervised_inds, ...]  # B_l,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'][supervised_inds, ...].long() # B_l,num_seed in [0,num_points-1]
    vote_label_mask = end_points['vote_label_mask'] # B, num_point
    vote_label = end_points['vote_label'] # B, num_point, 9

    seed_gt_votes_mask = torch.gather(vote_label_mask, 1,
                                                 seed_inds)  # B_l,num_seed
    seed_inds_expand = seed_inds.view(batch_size_supervised, num_seed, 1).expand(-1, -1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(vote_label, 1, seed_inds_expand)  # B_l,num_seed,9
    seed_gt_votes += seed_xyz.repeat(1, 1, 3)

    vote_xyz_reshape = vote_xyz.view(batch_size_supervised * num_seed, -1,
                                     3)  # from B_l,num_seed*vote_factor,3 to B_l*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size_supervised * num_seed, GT_VOTE_FACTOR,
                                             3)  # from B_l,num_seed,3*GT_VOTE_FACTOR to B_l*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B_l*num_seed,vote_factor) to (B_l*num_seed,)
    votes_dist = votes_dist.view(batch_size_supervised, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)

    return vote_loss


def compute_objectness_loss(end_points, supervised_inds):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """

    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'][supervised_inds, ...]
    gt_center = end_points['center_label'][:, :, 0:3]

    B = gt_center.shape[0] #B_l
    K = aggregated_vote_xyz.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center)  # dist1: B_l xK, dist2: B_l xK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B, K)).cuda()
    objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores'][supervised_inds, ...]
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2, 1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

    # Set assignment
    object_assignment = ind1  # (B_l ,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment


def compute_box_and_sem_cls_loss(end_points, supervised_inds, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    dist1, ind1, dist2, _ = nn_distance(end_points['center'][supervised_inds, ...], end_points['center_label'][:,:,0:3])  # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1,
                                       object_assignment)  # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'][supervised_inds, ...].transpose(2, 1),
                                                 heading_class_label)  # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1,
                                          object_assignment)  # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                   1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(
        torch.sum(end_points['heading_residuals_normalized'][supervised_inds, ...] * heading_label_one_hot,
                  -1) - heading_residual_normalized_label, delta=1.0)  # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'][supervised_inds, ...].transpose(2, 1), size_class_label)  # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1,
                                       object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized'][supervised_inds, ...] * size_label_one_hot_tiled,
                                                   2)  # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
        0)  # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
    size_residual_normalized_loss = torch.mean(
        huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0),
        -1)  # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label) / (
                torch.sum(objectness_label) + 1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'][supervised_inds, ...].transpose(2, 1), sem_cls_label)  # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss



def get_detection_loss(end_points, config):
    """ Loss functions for supervised samples in training detector

    Args:
        end_points: dict
            {
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        votenet_loss: pytorch scalar tensor
        end_points: dict
    """
    supervised_mask = end_points['supervised_mask']
    supervised_inds = torch.nonzero(supervised_mask).squeeze(1).long()

    # Vote loss
    vote_loss = compute_vote_loss(end_points, supervised_inds)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points, supervised_inds)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda()) / float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, supervised_inds, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    votenet_loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
    votenet_loss *= 10
    end_points['detection_loss'] = votenet_loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'][supervised_inds, ...], 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
            torch.sum(objectness_mask) + 1e-6)
    end_points['obj_acc'] = obj_acc

    return votenet_loss, end_points


def compute_center_consistency_loss(end_points, ema_end_points):
    center = end_points['center'] #(B, num_proposal, 3)
    ema_center = ema_end_points['center'] #(B, num_proposal, 3)
    flip_x_axis = end_points['flip_x_axis'] #(B,)
    flip_y_axis = end_points['flip_y_axis'] #(B,)
    rot_mat = end_points['rot_mat'] #(B,3,3)
    scale_ratio = end_points['scale'] #(B,1,3)

    # align ema_center with center based on the input augmentation steps
    inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
    ema_center[inds_to_flip_x_axis, :, 0] = -ema_center[inds_to_flip_x_axis, :, 0]

    inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
    ema_center[inds_to_flip_y_axis, :, 1] = -ema_center[inds_to_flip_y_axis, :, 1]

    ema_center = torch.bmm(ema_center, rot_mat.transpose(1,2)) #(B, num_proposal, 3)

    ema_center = ema_center * scale_ratio

    dist1, ind1, dist2, ind2 = nn_distance(center, ema_center)  #ind1 (B, num_proposal): ema_center index closest to center

    #TODO: use both dist1 and dist2 or only use dist1
    dist = dist1 + dist2
    return torch.mean(dist), ind2


def compute_class_consistency_loss(end_points, ema_end_points, map_ind):
    cls_scores = end_points['sem_cls_scores'] #(B, num_proposal, num_class)
    ema_cls_scores = ema_end_points['sem_cls_scores'] #(B, num_proposal, num_class)

    cls_log_prob = F.log_softmax(cls_scores, dim=2) #(B, num_proposal, num_class)
    # cls_log_prob = F.softmax(cls_scores, dim=2)
    ema_cls_prob = F.softmax(ema_cls_scores, dim=2) #(B, num_proposal, num_class)

    cls_log_prob_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(cls_log_prob, map_ind)])

    class_consistency_loss = F.kl_div(cls_log_prob_aligned, ema_cls_prob)
    # class_consistency_loss = F.mse_loss(cls_log_prob_aligned, ema_cls_prob)

    return class_consistency_loss*2


def compute_size_consistency_loss(end_points, ema_end_points, map_ind, config):
    mean_size_arr = config.mean_size_arr
    mean_size_arr = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda() #(num_size_cluster,3)
    B, K = map_ind.shape

    scale_ratio = end_points['scale'] #(B,1,3)
    size_class = torch.argmax(end_points['size_scores'], -1) # B,num_proposal
    size_residual = torch.gather(end_points['size_residuals'], 2, size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    size_residual.squeeze_(2)

    ema_size_class = torch.argmax(ema_end_points['size_scores'], -1) # B,num_proposal
    ema_size_residual = torch.gather(ema_end_points['size_residuals'], 2, ema_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,3)) # B,num_proposal,1,3
    ema_size_residual.squeeze_(2)

    size_base = torch.index_select(mean_size_arr, 0, size_class.view(-1))
    size_base = size_base.view(B,K,3)
    size = size_base + size_residual

    ema_size_base = torch.index_select(mean_size_arr, 0, ema_size_class.view(-1))
    ema_size_base = ema_size_base.view(B,K,3)
    ema_size = ema_size_base + ema_size_residual
    ema_size = ema_size * scale_ratio

    size_aligned = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(size, map_ind)])

    size_consistency_loss = F.mse_loss(size_aligned, ema_size)

    return size_consistency_loss


def get_consistency_loss(end_points, ema_end_points, config):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    center_consistency_loss, map_ind = compute_center_consistency_loss(end_points, ema_end_points)
    class_consistency_loss = compute_class_consistency_loss(end_points, ema_end_points, map_ind)
    size_consistency_loss = compute_size_consistency_loss(end_points, ema_end_points, map_ind, config)

    consistency_loss =  center_consistency_loss +class_consistency_loss + size_consistency_loss

    end_points['center_consistency_loss'] = center_consistency_loss
    end_points['class_consistency_loss'] = class_consistency_loss
    end_points['size_consistency_loss'] = size_consistency_loss
    end_points['consistency_loss'] = consistency_loss

    return consistency_loss, end_points






