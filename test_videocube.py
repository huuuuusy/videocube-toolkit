from __future__ import absolute_import

import os

from tracker.siamfc import TrackerSiamFC
from videocube.experiments import *

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # the path of VideoCube main folder
    root_dir = "/mnt/first/hushiyu/SOT/VideoCube/"

    version = 'tiny'  # set the version as 'tiny' or 'full'

    # the path to save the experiment result
    save_dir = os.path.join(root_dir, 'result')
    # the subset of VideoCube, please select train/test/val
    subset = 'test'
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

    """
    I. RUN TRACKER
    Note: 
    method in run function means the evaluation mechanism, you can select the original mode (set 'none') or the restart mode (set 'restart')
    """

    net_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'pretrained', 'siamfc', 'model.pth')
    tracker = TrackerSiamFC(net_path=net_path)

    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition + 1, version)
        experiment.run(tracker, visualize=False, save_img=False, method=None)

    """
    II. EVALUATION
    Note: 
    please set your tracker in first, then add the other trackers (you can download existing tracking results for 20 SOTA trackers via http://videocube.aitestunion.com/)
    """

    """evaluation in OPE"""
    tracker_names = ['SiamFC']
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition + 1, version)
        experiment.report(tracker_names, attribute_name='normal')

    """evaluation in R-OPE"""
    tracker_names = ['SiamFC_restart']
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition + 1, version)
        experiment.report(tracker_names, attribute_name='normal')
        experiment.report_robust(tracker_names)
