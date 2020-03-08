# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from utils import ioa_with_anchors, iou_with_anchors

from config.defaults import get_cfg


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


def train_collate_fn(batch):
    b_env_feats, b_agent_feats, b_lens, b_box_lens, confidence_labels, start_labels, end_labels = zip(*batch)

    # Make new order to inputs by their lengths (long-to-short)
    b_box_lens = torch.stack(b_box_lens, dim=0)

    max_box_dim = torch.max(b_box_lens).item()
    bsz = len(b_env_feats)
    max_tmp_dim, feat_dim = b_env_feats[0].size()

    len_descend = sorted(range(bsz), key=lambda x: b_lens[x], reverse=True)

    # Reorder inputs by new indices
    confidence_labels = torch.stack(confidence_labels)[len_descend]
    start_labels = torch.stack(start_labels)[len_descend]
    end_labels = torch.stack(end_labels)[len_descend]
    b_box_lens = b_box_lens[len_descend]
    b_env_feats = torch.stack(b_env_feats)[len_descend]
    b_agent_feats = [b_agent_feats[idx] for idx in len_descend]
    b_lens = [b_lens[idx] for idx in len_descend]

    # Make padding mask for self-attention
    b_agent_mask = torch.arange(max_box_dim)[None, None, :] >= b_box_lens[:, :, None]
    # print((torch.arange(max_temporal_dim)[None, :] > torch.tensor(batch_lengths)[:, None]).size())
    b_env_mask = torch.stack(
        [
            torch.arange(max_tmp_dim)[None, :] > torch.tensor(b_lens)[:, None],
            b_box_lens > 0
        ],
        dim=-1)

    # Pad agent features at temporal and box dimension
    pad_b_agent_feats = torch.zeros(bsz, max_tmp_dim, max_box_dim, feat_dim)
    for i, temporal_features in enumerate(b_agent_feats):
        for j, box_features in enumerate(temporal_features):
            if len(box_features) > 0:
                pad_b_agent_feats[i, j, :len(box_features)] = torch.tensor(box_features)

    return b_env_feats, pad_b_agent_feats, b_lens, b_env_mask, b_agent_mask, confidence_labels, start_labels, end_labels


def test_collate_fn(batch):
    video_names, b_env_feats, b_agent_feats, b_lens, b_box_lens = zip(*batch)

    # Make new order to inputs by their lengths (long-to-short)
    b_box_lens = torch.stack(b_box_lens, dim=0)

    max_box_dim = torch.max(b_box_lens).item()
    bsz = len(b_env_feats)
    max_tmp_dim, feat_dim = b_env_feats[0].size()

    len_descend = sorted(range(bsz), key=lambda x: b_lens[x], reverse=True)

    # Reorder inputs by new indices
    video_names = [video_names[idx] for idx in len_descend]
    b_box_lens = b_box_lens[len_descend]
    b_env_feats = torch.stack(b_env_feats)[len_descend]
    b_agent_feats = [b_agent_feats[idx] for idx in len_descend]
    b_lens = [b_lens[idx] for idx in len_descend]

    # Make padding mask for self-attention
    b_agent_mask = torch.arange(max_box_dim)[None, None, :] >= b_box_lens[:, :, None]
    # print((torch.arange(max_temporal_dim)[None, :] > torch.tensor(batch_lengths)[:, None]).size())
    b_env_mask = torch.stack(
        [
            torch.arange(max_tmp_dim)[None, :] > torch.tensor(b_lens)[:, None],
            b_box_lens > 0
        ],
        dim=-1)

    # Pad agent features at temporal and box dimension
    pad_b_agent_feats = torch.zeros(bsz, max_tmp_dim, max_box_dim, feat_dim)
    for i, temporal_features in enumerate(b_agent_feats):
        for j, box_features in enumerate(temporal_features):
            if len(box_features) > 0:
                pad_b_agent_feats[i, j, :len(box_features)] = torch.tensor(box_features)

    return video_names, b_env_feats, pad_b_agent_feats, b_lens, b_env_mask, b_agent_mask


class VideoDataSet(Dataset):
    def __init__(self, cfg, split="train"):
        self.split = split

        if split == 'train':
            self.video_anno_path = cfg.TRAIN.VIDEO_ANNOTATION_FILE
            self.temporal_dim = cfg.DATA.TEMPORAL_DIM
            self.temporal_gap = 1. / self.temporal_dim
            self.temporal_step = int(0.5 * self.temporal_dim)
            self._get_match_map()

        if self.split == 'validation':
            self.video_anno_path = cfg.VAL.VIDEO_ANNOTATION_FILE
            self.temporal_dim = None

        self.env_feature_dir = cfg.DATA.ENV_FEATURE_DIR
        self.agent_feature_dir = cfg.DATA.AGENT_FEATURE_DIR
        self.feature_lengths_path = cfg.DATA.FEATURE_LENGTHS_PATH
        self._get_dataset()

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_dim):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_dim + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map

        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_dim)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(1, self.temporal_dim + 1)]

    def _get_dataset(self):
        annotations = load_json(self.video_anno_path)
        self.video_names = list(annotations.keys())

        if os.path.isfile(self.feature_lengths_path):
            feature_lengths = load_json(self.feature_lengths_path)
        else:
            feature_lengths = {}
            for video_name in tqdm(self.video_names):
                num_features = load_json(
                    os.path.join(self.env_feature_dir, video_name + '.json')
                )['num_features']
                feature_lengths[video_name] = num_features

            with open(self.feature_lengths_path, 'w') as f:
                json.dump(feature_lengths, f)

        # Read event segments
        self.event_dict = {}

        if self.split == 'train':
            self.period_indices = []

        print('Reading dataset.')
        for video_name in tqdm(self.video_names):
            annotation = annotations[video_name]
            num_features = feature_lengths[video_name]

            self.event_dict[video_name] = {
                'num_features': num_features,
                'events': annotation['timestamps']
            }

            if self.split == 'train':
                if num_features + 1 > self.temporal_step:
                    for start_idx in range(0, num_features - self.temporal_step + 1, self.temporal_step):
                        self.period_indices.append({'video_name': video_name, 'start_idx': start_idx})
                else:
                    self.period_indices.append({'video_name': video_name, 'start_idx': 0})

        if self.split == 'train':
            print("Split: %s. Dataset size: %d" % (self.split, len(self.period_indices)))
        else:
            print("Split: %s. Dataset size: %d" % (self.split, len(self.video_names)))

    def __getitem__(self, index):
        env_features, agent_features, env_length, box_lengths, feature_period = self._load_item(index)
        if self.split == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, feature_period)
            return env_features, agent_features, env_length, box_lengths, confidence_score, match_score_start, match_score_end
        else:
            return self.video_names[index], env_features, agent_features, env_length, box_lengths

    def _load_item(self, index):
        if self.split == 'train':
            video_name, start_idx = self.period_indices[index].values()
        else:
            video_name = self.video_names[index]

        '''
        Read environment features at every timestamp
        Feature size: TxF
        T: number of timestamps
        F: feature size
        '''
        env_features = load_json(os.path.join(self.env_feature_dir, video_name + '.json'))['video_features']
        if self.split == 'train':
            env_features = env_features[start_idx:start_idx + self.temporal_dim]
        env_segments = [env['segment'] for env in env_features]
        env_features = torch.tensor([feature['features'] for feature in env_features]).float().squeeze(1)

        env_length = len(env_features)

        # Pad environment features if train
        if self.split == 'train':
            tmp_dim, feat_dim = env_features.size()
            env_features = torch.cat([env_features, torch.zeros(self.temporal_dim - tmp_dim, feat_dim)], dim=0)

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        agent_features = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))['video_features']
        if self.split == 'train':
            agent_features = agent_features[start_idx:start_idx + self.temporal_dim]
        agent_segments = [feature['segment'] for feature in agent_features]
        agent_features = [feature['features'] for feature in agent_features]

        assert env_segments == agent_segments, 'Two streams must have same segments.'

        # Merge agent and environment features
        # merge_features = []
        # for env_feature, agent_feature in zip(env_features, agent_features):
        #     merge_features.append(env_feature + agent_feature)

        # Create and pad agent_box_lengths if train
        box_lengths = torch.tensor([len(x) for x in agent_features])
        if self.split == 'train':
            box_lengths = torch.cat([box_lengths, torch.zeros(self.temporal_dim - len(agent_features)).long()], dim=0)

        begin_timestamp, end_timestamp = env_segments[0][0], env_segments[-1][-1]
        return env_features, agent_features, env_length, box_lengths, (begin_timestamp, end_timestamp)

    def _get_train_label(self, index, period):
        video_name = self.period_indices[index]['video_name']
        video_info = self.event_dict[video_name]
        video_labels = video_info['events']  # the measurement is second, not frame

        if period[1] < period[0]:
            period[1], period[0] = period
        duration = period[1] - period[0]

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = (np.clip(tmp_info[0], period[0], period[1]) - period[0]) / duration
            tmp_end = (np.clip(tmp_info[0], period[0], period[1]) - period[0]) / duration
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_dim, self.temporal_dim])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ##############################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        ##############################################################################################

        ##############################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.tensor(match_score_start)
        match_score_end = torch.tensor(match_score_end)
        ##############################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        if self.split == 'train':
            return len(self.period_indices)
        else:
            return len(self.video_names)


if __name__ == '__main__':
    cfg = get_cfg()
    train_loader = torch.utils.data.DataLoader(VideoDataSet(cfg, split="train"),
                                               batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                               num_workers=8, pin_memory=True, collate_fn=train_collate_fn)
    for a, b, c, d, e, f in train_loader:
        print(a.size(), b.size(), c.size(), d.size(), e.size(), f.size())
        break
