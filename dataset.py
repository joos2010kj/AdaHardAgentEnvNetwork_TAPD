# -*- coding: utf-8 -*-
import os
import json
import math

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from utils import ioa_with_anchors, iou_with_anchors

from config.defaults import get_cfg


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


def train_collate_fn(batch):
    batch_env_features, batch_agent_features, batch_agent_box_lengths, confidence_labels, start_labels, end_labels = zip(*batch)

    confidence_labels = torch.stack(confidence_labels)
    start_labels = torch.stack(start_labels)
    end_labels = torch.stack(end_labels)

    # Create agent feature padding mask
    batch_agent_box_lengths = torch.stack([
        torch.cat([
            agent_box_lengths,
            torch.zeros(cfg.TEMPORAL_DIM - len(agent_box_lengths))
        ], dim=0)
        for agent_box_lengths in batch_agent_box_lengths
    ])
    max_box_dim = torch.max(batch_agent_box_lengths).item()
    batch_agent_padding_mask = torch.arange(max_box_dim)[None, None, :] < batch_agent_box_lengths[:, :, None]

    # Pad environment features at temporal dimension
    batch_env_features = torch.stack([
        torch.cat([
            env_features,
            torch.zeros(cfg.DATA.TEMPORAL_DIM - len(env_features), cfg.DATA.FEATURE_DIM)
        ])
        for env_features in batch_env_features
    ])

    # Pad agent features at temporal and box dimension
    batch_size, max_temporal_dim, feature_dim = batch_env_features.size()
    padded_batch_agent_features = torch.zeros(batch_size, max_temporal_dim, max_box_dim, feature_dim)

    for i, temporal_features in enumerate(batch_agent_features):
        for j, box_features in enumerate(temporal_features):
            if len(box_features) > 0:
                padded_batch_agent_features[i, j, :len(box_features)] = torch.tensor(box_features)

    return batch_env_features, padded_batch_agent_features, batch_agent_padding_mask, confidence_labels, start_labels, end_labels


def test_collate_fn(batch):
    indices, batch_env_features, batch_agent_features, batch_agent_box_lengths = zip(*batch)

    # Create agent feature padding mask
    batch_agent_box_lengths = pad_sequence(batch_agent_box_lengths, batch_first=True)
    max_box_dim = torch.max(batch_agent_box_lengths).item()
    batch_agent_padding_mask = torch.arange(max_box_dim)[None, None, :] < batch_agent_box_lengths[:, :, None]

    # Pad environment features at temporal dimension
    batch_env_features = pad_sequence(batch_env_features, batch_first=True)

    # Pad agent features at temporal and box dimension
    batch_size, max_temporal_dim, feature_dim = batch_env_features.size()
    padded_batch_agent_features = torch.zeros(batch_size, max_temporal_dim, max_box_dim, feature_dim)

    for i, temporal_features in enumerate(batch_agent_features):
        for j, box_features in enumerate(temporal_features):
            if len(box_features) > 0:
                padded_batch_agent_features[i, j, :len(box_features)] = torch.tensor(box_features)

    return indices, batch_env_features, padded_batch_agent_features, batch_agent_padding_mask


class VideoDataSet(Dataset):
    def __init__(self, cfg, split="train"):
        self.split = split
        self.env_feature_dir = cfg.DATA.ENV_FEATURE_DIR
        self.agent_feature_dir = cfg.DATA.AGENT_FEATURE_DIR
        self.video_id_path = cfg.DATA.VIDEO_ID_FILE
        self.video_anno_path = cfg.DATA.VIDEO_ANNOTATION_FILE

        self.temporal_dim = cfg.DATA.TEMPORAL_DIM
        self.temporal_gap = 1. / self.temporal_dim
        self.temporal_step = int(0.5 * self.temporal_dim)

        self.target_fps = cfg.DATA.TARGET_FPS
        self.sampling_rate = cfg.DATA.SAMPLING_RATE

        self._get_dataset(cfg)
        self._get_match_map()

        if self.split != 'train':
            self.temporal_dim = None

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

    def _get_dataset(self, cfg):
        self.video_names = load_json(self.video_id_path)
        annotations = load_json(self.video_anno_path)
        # Read event segments
        self.event_dict = {}

        if self.split == 'train':
            self.period_indices = []

        for video_name in self.video_names:
            annotation = annotations[video_name]
            self.event_dict[video_name] = {
                'duration': annotation['duration'],
                'events': annotation['timestamps']
            }

            estimate_num_features = math.floor(annotation['duration'] * self.target_fps / self.sampling_rate)

            if self.split == 'train':
                for start_idx in range(0, estimate_num_features - self.temporal_step + 1, self.temporal_step):
                    self.period_indices.append({'video_name': video_name, 'start_idx': start_idx})

        print("Split: %s. Video numbers: %d" % (self.split, len(self.video_names)))

    def __getitem__(self, index):
        env_features, agent_features, agent_box_lengths, feature_period = self._load_item(index)
        if self.split == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, feature_period)
            return env_features, agent_features, agent_box_lengths, confidence_score, match_score_start, match_score_end
        else:
            return index, env_features, agent_features, agent_box_lengths

    def _load_item(self, index):
        if self.split == 'train':
            video_name, start_idx = self.period_indices[index].values()
        else:
            video_name = self.video_names[index]
            start_idx = 0

        '''
        Read environment features at every timestamp
        Feature size: TxF
        T: number of timestamps
        F: feature size
        '''
        env_features_dict = load_json(os.path.join(self.env_feature_dir, video_name + '.json'))

        env_indices = sorted(env_features_dict.keys(), key=lambda x: float(x))
        env_indices = env_indices[start_idx:self.temporal_dim]
        env_segments = [env_features_dict[t]['segment'] for t in env_indices]

        env_features = torch.tensor([env_features_dict[t]['features'] for t in env_indices]).float().squeeze(1)

        '''
        Read agents features at every timestamp
        Feature size: TxBxF
        T: number of timestamps
        B: max number of bounding boxes
        F: feature size
        '''
        agent_features_dict = load_json(os.path.join(self.agent_feature_dir, video_name + '.json'))

        agent_indices = sorted(agent_features_dict.keys(), key=lambda x: float(x))
        agent_indices = agent_indices[start_idx:self.temporal_dim]
        agent_segments = [agent_features_dict[t]['segment'] for t in agent_indices]
        agent_features = [agent_features_dict[t]['features'] for t in agent_indices]

        agent_box_lengths = [len(x) for x in agent_features]

        assert env_segments == agent_segments, 'Two streams must have same paces.'
        begin_timestamp, end_timestamp = env_segments[0][0], env_segments[-1][-1]

        return env_features, agent_features, agent_box_lengths, (begin_timestamp, end_timestamp)

    def _get_train_label(self, index, period):
        video_name = self.video_names[index]
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
                                        [math.floor(duration), math.floor(duration)])
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
    for a, b, c, d, e in train_loader:
        print(a.size(), b.size(), c.size(), d.size(), e.size())
        break
