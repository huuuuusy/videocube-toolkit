from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2 as cv

from ..datasets import GOT10k
from ..utils.metrics import rect_iou
from ..utils.viz import show_frame
from ..utils.ioutils import compress
from ..utils.help import makedir


class ExperimentGOT10k(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.
    
    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        subset (string): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        list_file (string, optional): If provided, only run experiments on
            sequences specified by this file.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, save_dir, subset, repetition, list_file=None, use_dataset=True):
        super(ExperimentGOT10k, self).__init__()
        # assert subset in ['val', 'test']
        self.subset = subset
        if use_dataset:
            self.dataset = GOT10k(
                root_dir, subset=subset, list_file=list_file)
        self.result_dir = os.path.join(save_dir, 'results') 
        self.report_dir = os.path.join(save_dir, 'reports') 
        self.time_dir = os.path.join(save_dir, 'time')
        self.analysis_dir = os.path.join(save_dir, 'analysis')
        self.img_dir = os.path.join(save_dir, 'image')

        self.nbins_iou = 101
        
        self.repetition = repetition 
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.time_dir)
        makedir(self.analysis_dir)
        makedir(self.img_dir)

    def run(self, tracker, visualize, save_img, method):
        if self.subset == 'test':
            print('\033[93m[WARNING]:\n' \
                  'The groundtruths of GOT-10k\'s test set is withholded.\n' \
                  'You will have to submit your results to\n' \
                  '[http://got-10k.aitestunion.com/]' \
                  '\nto access the performance.\033[0m')
            time.sleep(2)

        print('Running tracker %s on GOT-10k...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno, restart_flag) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetition):
                # check if the tracker is deterministic
                print(' Repetition: %d'%self.repetition)
                if method == None:
                # tracking in OPE mechanism
                    record_name = tracker.name
                else:
                    # tracking in R-OPE mechanism
                    record_name = tracker.name + '_' + method

                makedir(os.path.join(self.result_dir, record_name))
                makedir(os.path.join(self.time_dir, record_name))

                tracker_result_dir = os.path.join(self.result_dir, record_name, self.subset)
                tracker_time_dir = os.path.join(self.time_dir, record_name, self.subset)

                makedir(tracker_result_dir)                
                makedir(tracker_time_dir)

                # setting the dir for saving tracking result images
                makedir( os.path.join(self.img_dir, record_name))
                tracker_img_dir = os.path.join(self.img_dir, record_name, self.subset)
                makedir(tracker_img_dir)
                seq_result_dir = os.path.join(tracker_img_dir, seq_name)
                makedir(seq_result_dir)

                # setting the path for saving tracking result
                record_file = os.path.join(tracker_result_dir, '%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))

                if os.path.exists(record_file):
                    print('  Found results, skipping ', seq_name)
                    continue

                # setting the path for saving tracking result (restart position in R-OPE mechanism)
                init_positions_file = os.path.join(tracker_result_dir, 'init_%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))

                # setting the path for saving tracking time 
                time_file = os.path.join(tracker_time_dir, '%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))
            
                # tracking loop
                if method == None:
                    # tracking in original OPE mechanism
                    boxes, times = tracker.track(seq_name, img_files, anno, restart_flag, visualize, seq_result_dir, save_img, method)
                elif method == 'restart':
                    # tracking in novel R-OPE mechanism
                    boxes, times, init_positions = tracker.track(seq_name, img_files, anno,  restart_flag, visualize, seq_result_dir, save_img, method)
                    # save the restart locations
                    f_init = open(init_positions_file, 'w')
                    for num in init_positions:
                        f_init.writelines(str(num)+'\n')
                    f_init.close()

                self._record(record_file, time_file, boxes, times)

    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        if self.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = os.path.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                compress('.', save_file)
                print('Records saved at', save_file + '.zip')

            # print submission guides
            print('\033[93mLogin and follow instructions on')
            print('http://got-10k.aitestunion.com/submit_instructions')
            print('to upload and evaluate your tracking results\033[0m')

            # switch back to previous working directory
            os.chdir(pwd)

            return None
        elif self.subset == 'val':
            # meta information is useful when evaluation
            self.dataset.return_meta = True

            # assume tracker_names[0] is your tracker
            report_dir = os.path.join(self.report_dir, tracker_names[0])
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names
            covers = {s: self.dataset[s][2]['cover'][1:] for s in seq_names}
            
            performance = {}
            for name in tracker_names:
                print('Evaluating', name)
                ious = {}
                times = {}
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

                for s, (_, anno, meta) in enumerate(self.dataset):
                    seq_name = self.dataset.seq_names[s]
                    record_files = glob.glob(os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_[0-9]*.txt' % seq_name))
                    if len(record_files) == 0:
                        raise Exception('Results for sequence %s not found.' % seq_name)

                    # read results of all repetitions
                    boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                    assert all([b.shape == anno.shape for b in boxes])

                    # calculate and stack all ious
                    bound = ast.literal_eval(meta['resolution'])
                    seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]
                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ious = np.concatenate(seq_ious)
                    ious[seq_name] = seq_ious

                    # stack all tracking times
                    times[seq_name] = []
                    time_file = os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_time.txt' % seq_name)
                    if os.path.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                ious = np.concatenate(list(ious.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, succ_curve = self._evaluate(ious, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})
            
            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            # plot success curves
            if plot_curves:
                self.plot_curves([report_file], tracker_names)

            return performance

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))
        
        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')
            
            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, time_file, boxes, times):
        """记录跟踪结果"""
        np.savetxt(record_file, boxes, fmt='%d', delimiter=',')
        print('Results recorded at', record_file)

        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot'+extension)
        key = 'overall'
        
        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]
        
        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))
        
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()
        
        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
