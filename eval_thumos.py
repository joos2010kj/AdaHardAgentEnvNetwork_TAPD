# -*- coding: utf-8 -*-
import os
import requests
import pickle
import io

import pandas as pd
from scipy.interpolate import interp1d

from evaluation_thumos import prop_eval


def run_evaluation(proposal_filename, groundtruth_filename='../datasets/thumos14/thumos14_test_groundtruth.csv'):
    frm_nums = pickle.load(open("evaluation_thumos/frm_num.pkl", 'rb'))
    rows = prop_eval.pkl2dataframe(frm_nums, 'evaluation_thumos/movie_fps.pkl', proposal_filename)
    aen_results = pd.DataFrame(rows, columns=['f-end', 'f-init', 'score', 'video-frames', 'video-name'])

    # Retrieves and loads Thumos14 test set ground-truth.
    if not os.path.isfile(groundtruth_filename):
        ground_truth_url = ('https://gist.githubusercontent.com/cabaf/'
                            'ed34a35ee4443b435c36de42c4547bd7/raw/'
                            '952f17b9cdc6aa4e6d696315ba75091224f5de97/'
                            'thumos14_test_groundtruth.csv')
        s = requests.get(ground_truth_url).content
        groundtruth = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=' ')
        groundtruth.to_csv(groundtruth_filename)
    else:
        groundtruth = pd.read_csv(groundtruth_filename)
    # Computes recall for different tiou thresholds at a fixed average number of proposals.
    '''
    recall, tiou_thresholds = prop_eval.recall_vs_tiou_thresholds(aen_results, ground_truth,
                                                        nr_proposals=nr_proposals,
                                                        tiou_thresholds=np.linspace(0.5, 1.0, 11))
    recall = np.mean(recall)
    '''
    average_recall, average_nr_proposals = prop_eval.average_recall_vs_nr_proposals(aen_results, groundtruth)

    return average_recall, average_nr_proposals


def evaluate_proposals(cfg, nr_proposals_list=(50, 100, 200, 500, 1000)):
    average_recall, average_nr_proposals = run_evaluation(cfg.DATA.RESULT_PATH)
    f = interp1d(average_nr_proposals, average_recall, axis=0, bounds_error=False, fill_value='extrapolate')

    ar_results = {}
    for nr_prop in nr_proposals_list:
        ar_results[nr_prop] = float(f(nr_prop))
        print("AR@{} is {}\n".format(nr_prop, ar_results[nr_prop]))

    return ar_results[100]


'''
    average_recalls = {}
    for nr_proposals in nr_proposals_list:
        average_recalls[nr_proposals] = run_evaluation(cfg.DATA.RESULT_PATH, nr_proposals)
    for nr_proposals, average_recall in average_recalls.items():
        print("AR@%d is %f\t" % (nr_proposals, average_recall))
    return average_recalls[100]
'''
