from __future__ import absolute_import

from videocube.experiments import *

from tracker.siamfc import TrackerSiamFC


if __name__ == '__main__':
    # the path of VideoCube data folder
    root_dir = "/mnt/first/hushiyu/VIS/data/"
    # the path to save the experiment result
    save_dir = "/mnt/first/hushiyu/VIS/VIS-result/"
    # the subset of VideoCube, please select train/test/val
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

    # I. RUN
    net_path = '/home/user1/projects/VIS/videocube-toolkit-official/pretrained/siamfc/model.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
        # 'the evaluation mechanism, you can select the original mode (set none) or the restart mode (set restart)'
        experiment.run(tracker, visualize=False, save_img=False, method='restart')


    # II. EVALUATION

    # please set your tracker in first, then add the other trackers (you can download existing tracking results for 20 SOTA trackers via http://videocube.aitestunion.com/)
    tracker_names = ['SiamFC']

    # 1. evaluation in OPE
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
        experiment.report(tracker_names, attribute_name='normal', eye_mode=False)
    
    # # TODO. comprehensive normal evaluation
    # experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetitions)
    # experiment.report_all(tracker_names, attribute_name='normal', eye_mode=False)


    # 2. evaluation in R-OPE
    tracker_names = ['SiamFC_restart']
    for repetition in range(repetitions):
        experiment = ExperimentVideoCube(root_dir, save_dir, subset, repetition+1)
        experiment.report(tracker_names, attribute_name='normal', eye_mode=False)
        experiment.report_robust(root_dir, tracker_names)
        




