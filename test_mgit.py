from __future__ import absolute_import

import os

from mgit.experiments import *

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # path for dataset and tracker_results
    # the path of MGIT dataset
    dataset_dir = "/path_to_MGIT"
    # the path of tracker e.g. JointNLT results
    root_dir = "/path_to_tracker_result"

    # temporarily, the toolkit only support tiny version of MGIT
    version = 'tiny'
    # the subset of MGIT, please select train/test/val
    subset = 'val'

    # the path to save the experiment result
    save_dir = os.path.join(root_dir, 'result')

    repetitions = 1

    attribute_list = [
        'delta_blur',
        'color_constancy_tran',
        'delta_color_constancy_tran',
        'corrcoef',
        'ratio',
        'delta_ratio',
        'scale',
        'delta_scale',
        'motion',
        'occlusion',
    ]

    # """
    # I. CONVERT RESULTS TO MGIT FORMAT
    # Note:
    # convert tracking results of algorithms based on pytracking framework e.g. JointNLT to MGIT format
    # """

    # tracker name
    tracker_name = "jointnlt"
    # original result folder name
    original_results_folder = "swin_b_ep300"

    # covert results to MGIT format
    for repetition in range(repetitions):
        experiment = ExperimentMGIT(dataset_dir, save_dir, subset, repetition + 1, version)
        experiment.convert_results(root_dir, tracker_name, original_results_folder)

    # """ II. EVALUATION Note: please set your tracker in first, then add the other trackers (you can download
    # existing tracking results for 20 SOTA trackers via http://videocube.aitestunion.com/) """

    # """evaluation in OPE"""
    tracker_names = ['jointnlt']
    for repetition in range(repetitions):
        experiment = ExperimentMGIT(dataset_dir, save_dir, subset, repetition + 1, version)
        experiment.report(tracker_names, attribute_name='normal')
