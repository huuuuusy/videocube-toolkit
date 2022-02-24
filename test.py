from __future__ import absolute_import

from videocube.experiments import *

from tracker.siamfc import TrackerSiamFC
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # the path of VideoCube data folder
    root_dir = "/mnt/first/hushiyu/SOT/VideoCube/"
    # root_dir = "/mnt/second/hushiyu/"
    # the path to save the experiment result
    save_dir = os.path.join(root_dir,'result')
    # the subset of VideoCube, please select train/test/val/eye
    subset = 'val'
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
    method in run function means the evaluation mechanism, you can select the original mode (set none) or the restart mode (set restart)
    """ 
    
    net_path = '/home/user1/projects/VIS/videocube-toolkit-official/pretrained/siamfc/model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    # for repetition in range(repetitions):
    #     experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
    #     experiment.run(tracker, visualize=False, save_img=False, method='restart')


    """
    II. EVALUATION
    Note: 
    please set your tracker in first, then add the other trackers (you can download existing tracking results for 20 SOTA trackers via http://videocube.aitestunion.com/)
    """
    tracker_names = ['SiamFC']

    """evaluation in OPE"""
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
        experiment.report(tracker_names, attribute_name='normal')

    """evaluation in R-OPE"""
    tracker_names = ['SiamFC_restart']
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
        experiment.report(tracker_names, attribute_name='normal')
        experiment.report_robust(root_dir, tracker_names)

    """evaluation in eye tracking subset"""
    tracker_names = ['SiamFC']
    subset = 'eye'
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset,repetition+1)
        experiment.eye_report(root_dir,tracker_names)
