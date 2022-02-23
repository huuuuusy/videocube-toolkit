from __future__ import absolute_import, division, print_function

import os
from time import time
import numpy as np

import json
import matplotlib.pyplot as plt
import matplotlib

from ..datasets import VideoCube
from ..utils.metrics import center_error,normalized_center_error, iou, diou, giou
from ..utils.ioutils import compress
from ..utils.help import makedir
import cv2 as cv
import pandas as pd
import seaborn as sns
from collections import defaultdict

class ExperimentVideoCube(object):
    r"""Experiment pipeline and evaluation toolkit for VideoCube dataset.
    
    Args:
        root_dir (string): 
            Root directory of VideoCube dataset where ``train``, ``val`` and ``test`` folders exist.
        save_dir (string): 
            Save directory of VideoCube dataset to save the experiment results.
        subset (string): 
            Specify ``train``, ``val`` or ``test`` subset of VideoCube.
        repetition (int): 
            The num of repetition. To ensure the accuracy of the experimental results, it is generally repeated three times.
    """
    def __init__(self, root_dir, save_dir, subset, repetition):
        super(ExperimentVideoCube, self).__init__()
        self.root_dir = root_dir
        self.subset = subset
        self.dataset = VideoCube(root_dir, subset)
        self.result_dir = os.path.join(save_dir, 'results') 
        self.report_dir = os.path.join(save_dir, 'reports') 
        self.time_dir = os.path.join(save_dir, 'time')
        self.analysis_dir = os.path.join(save_dir, 'analysis')
        self.img_dir = os.path.join(save_dir, 'image')
        
        self.nbins_iou = 101 # set 101 points in drawing success plot
        self.nbins_ce = 401 # set 401 points in drawing original precision plot (the 401 is the top threshold value in calculating the PRE)

        self.ce_threshold = 20 # original precision plot selects 20 pixels as threshold

        self.repetition = repetition 
        makedir(save_dir)
        makedir(self.result_dir)
        makedir(self.report_dir)
        makedir(self.time_dir)
        makedir(self.analysis_dir)
        makedir(self.img_dir)
        

    def run(self, tracker, visualize, save_img, method):
        """
        Run the tracker on VideoCube subset.
        """
        print('Running tracker %s on VideoCube...' % tracker.name)
        print('evaluation mechanism: %s' % method)

        for s, (img_files, anno, restart_flag) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s] 
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            print('  Repetition: %d'%self.repetition)
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

            print(record_file)

            # setting the path for saving tracking result (restart position in R-OPE mechanism)
            init_positions_file = os.path.join(tracker_result_dir, 'init_%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))

            # setting the path for saving tracking time 
            time_file = os.path.join(tracker_time_dir, '%s_%s_%s.txt'%(record_name , seq_name , str(self.repetition)))
            
            if os.path.exists(record_file):
                print('  Found results, skipping ', seq_name)
                continue
            
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


    def report(self, tracker_names, attribute_name):
        """
        Evaluate the tracker on VideoCube subset.
        """
        assert isinstance(tracker_names, (list, tuple))
        
        subset_report_dir = os.path.join(self.report_dir, self.subset)
        makedir(subset_report_dir)

        subset_analysis_dir = os.path.join(self.analysis_dir, self.subset)
        makedir(subset_analysis_dir)

        report_dir = os.path.join(subset_report_dir, tracker_names[0])
        makedir(report_dir)

        performance = {}
        for name in tracker_names:

            single_report_file = os.path.join(subset_analysis_dir, '%s_%s_%s_%s.json'%(name, self.subset, attribute_name, str(self.repetition)))
            
            if os.path.exists(single_report_file):
                f = open(single_report_file,'r',encoding='utf-8')
                single_performance = json.load(f)            
                performance.update({name:single_performance})
                f.close()
                print('Existing result in {}'.format(name))
                continue
            else:
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})
            
            seq_num = len(self.dataset)

            # save the ious, dious and gious for success plot
            succ_curve = np.zeros((seq_num, self.nbins_iou))
            succ_dcurve = np.zeros((seq_num, self.nbins_iou))
            succ_gcurve = np.zeros((seq_num, self.nbins_iou))

            # save the original precision value for original precision plot
            prec_curve = np.zeros((seq_num, self.nbins_ce))
            # save the novel precision value for normalized precision plot
            norm_prec_curve = np.zeros((seq_num, self.nbins_ce))

            # save average speed for each video
            speeds = np.zeros(seq_num)

            # save the normalize precision score
            norm_prec_score  = np.zeros(seq_num)
            
            for s in range(len(self.dataset.seq_names)):
                num = self.dataset.seq_names[s]
                # get the information of selected video
                img_files, anno, _ = self.dataset[s]

                print('repetition %s: Evaluate tracker %s in video num %s with %s attribute '%(self.repetition, name, num, attribute_name))
                
                # read present info
                present_path = os.path.join(self.root_dir, self.subset, num,'present_%s.txt'%num)
                present = pd.read_csv(present_path,header=None,names=['present'])

                # read shotcut info
                shotcut_path = os.path.join(self.root_dir, self.subset, num,'shotcut_%s.txt'%num)
                shotcut = pd.read_csv(shotcut_path,header=None,names=['shotcut'])

                # read attribute info
                if attribute_name != 'normal':
                    attribute_path = os.path.join(self.root_dir, self.subset, num,'%s_%s.txt'%(attribute_name, num))
                    attribute = pd.read_csv(attribute_path,header=None,names=[attribute_name])
                
                # frame resolution
                img_height = cv.imread(img_files[0]).shape[0]
                img_width = cv.imread(img_files[0]).shape[1]
                img_resolution = (img_width,img_height)
                bound = img_resolution

                # read tracking results
                boxes = np.loadtxt(os.path.join(self.result_dir, name, self.subset, '{}_{}_{}.txt'.format(name, num, self.repetition)), delimiter=',')

                anno = np.array(anno)
                boxes = np.array(boxes)

                for box in boxes:
                    # correction of out-of-range coordinates
                    box[0] = box[0] if box[0] > 0 else 0
                    box[2] = box[2] if box[2] < img_width - box[0] else img_width - box[0]
                    box[1] = box[1] if box[1] > 0 else 0
                    box[3] = box[3] if box[3] < img_height - box[1] else img_height - box[1]

                assert boxes.shape == anno.shape
                
                # calculate ious, gious, dious for success plot
                # calculate center errors and normalized center errors for precision plot
                seq_ious, seq_dious, seq_gious, seq_center_errors, seq_norm_center_errors, flags = self._calc_metrics(boxes, anno, bound)
                
                seq_ious = pd.DataFrame(seq_ious, columns = ['seq_ious'])
                seq_dious = pd.DataFrame(seq_dious, columns = ['seq_dious'])
                seq_gious = pd.DataFrame(seq_gious, columns = ['seq_gious'])
                seq_center_errors = pd.DataFrame(seq_center_errors, columns = ['seq_center_errors'])
                seq_norm_center_errors = pd.DataFrame(seq_norm_center_errors, columns= ['seq_norm_center_errors'])
                flags = pd.DataFrame(flags, columns = ['flags'])

                if attribute_name != 'normal':
                    data = pd.concat([seq_ious,seq_dious, seq_gious,seq_center_errors,seq_norm_center_errors,flags, present,shotcut,attribute],axis=1) 
                else:
                    data = pd.concat([seq_ious,seq_dious, seq_gious,seq_center_errors,seq_norm_center_errors,flags, present,shotcut],axis=1) 
                
                # Frames without target and transition frames are not included in the evaluation
                data = data[data.apply(lambda x: x['present']==1 and x['shotcut']==0, axis=1)]

                if attribute_name != 'normal':
                    # Pick frames with difficult attributes
                    data = self.select_attribute_frames(data, attribute_name)  
                
                data = data.drop(labels=['present','shotcut'],axis = 1)

                seq_ious = data['seq_ious']
                seq_dious = data['seq_dious']
                seq_gious = data['seq_gious']
                seq_center_errors = data['seq_center_errors']
                seq_norm_center_errors = data['seq_norm_center_errors']   
                flags = data['flags']  
                
                # Calculate the proportion of all the frames that fall into area 5 (groundtruth area)
                norm_prec_score[s] = np.nansum(flags)/len(flags)

                # Save the 5 curves of the tracker on the current video
                succ_curve[s], succ_dcurve[s], succ_gcurve[s],prec_curve[s], norm_prec_curve[s] = self._calc_curves(seq_ious, seq_dious, seq_gious,seq_center_errors, seq_norm_center_errors)

                # calculate average speed
                time_file = os.path.join(
                    self.time_dir, name, self.subset,'{}_{}_{}.txt'.format(name, num, self.repetition)) 
                
                if os.path.isfile(time_file):
                    times = np.loadtxt(time_file)
                    
                    times = times[times > 0]
                    if len(times) > 0:
                        speeds[s] = np.nanmean(1. / times)

                # Update the results in current video (Only save scores)
                performance[name]['seq_wise'].update({num: {
                    'success_score_iou': np.nanmean(succ_curve[s]),
                    'success_score_diou': np.nanmean(succ_dcurve[s]),                    
                    'success_score_giou': np.nanmean(succ_gcurve[s]),
                    'precision_score': prec_curve[s][self.ce_threshold],
                    'norm_prec_score':norm_prec_score[s],
                    'success_rate_iou': succ_curve[s][self.nbins_iou // 2],
                    'success_rate_diou': succ_dcurve[s][self.nbins_iou // 2],
                    'success_rate_giou': succ_gcurve[s][self.nbins_iou // 2],
                    'speed_fps': speeds[s] if speeds[s] > 0 else -1}})

            # Average each curve
            succ_curve = np.nanmean(succ_curve, axis=0)
            succ_dcurve = np.nanmean(succ_dcurve, axis=0)
            succ_gcurve = np.nanmean(succ_gcurve, axis=0)
            prec_curve = np.nanmean(prec_curve, axis=0)
            norm_prec_curve = np.nanmean(norm_prec_curve, axis=0)

            # Generate average score
            succ_score = np.nanmean(succ_curve)
            succ_dscore = np.nanmean(succ_dcurve)
            succ_gscore = np.nanmean(succ_gcurve)
            succ_rate = succ_curve[self.nbins_iou // 2]
            succ_drate = succ_dcurve[self.nbins_iou // 2]
            succ_grate = succ_gcurve[self.nbins_iou // 2]

            prec_score = prec_curve[self.ce_threshold]
            norm_prec_score = np.nansum(norm_prec_score) / np.count_nonzero(norm_prec_score)

            if np.count_nonzero(speeds) > 0:
                avg_speed = np.nansum(speeds) / np.count_nonzero(speeds)
            else:
                avg_speed = -1

            # store overall performance
            performance[name]['overall'].update({
                'success_curve_iou': succ_curve.tolist(),
                'success_curve_diou': succ_dcurve.tolist(),
                'success_curve_giou': succ_gcurve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'success_score_iou': succ_score,
                'success_score_diou': succ_dscore,
                'success_score_giou': succ_gscore,
                'precision_score': prec_score,
                'norm_prec_score':norm_prec_score,
                'success_rate_iou': succ_rate,
                'success_rate_diou': succ_drate,
                'success_rate_giou': succ_grate,
                'speed_fps': avg_speed})

            with open(single_report_file, 'w') as f:
                json.dump(performance[name], f, indent=4)

        # save performance
        report_file = os.path.join(report_dir, '{}_performance_{}.json'.format(attribute_name, str(self.repetition)))
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)

        self.plot_curves_([report_file], tracker_names, attribute_name, self.repetition)

        return performance
    

    def select_attribute_frames(self, data, attribute_name):
        """
        Pick frames with difficult attributes
        """
        if attribute_name == 'corrcoef':
            data = data[data.apply(lambda x: x[attribute_name] <=0.8, axis=1)]
        elif attribute_name == 'motion':
            data = data[data.apply(lambda x: x[attribute_name] >=0.2, axis=1)]
        elif attribute_name == 'occlusion':
            data = data[data.apply(lambda x: x[attribute_name] == 1, axis=1)]
        elif attribute_name == 'delta_blur':
            data = data[data.apply(lambda x: x[attribute_name] >= 1.5, axis=1)] 
        elif attribute_name == 'ratio':
            data = data[data.apply(lambda x: x[attribute_name] >=3 or x[attribute_name] <=0.33, axis=1)] 
        elif attribute_name == 'delta_ratio':
            data = data[data.apply(lambda x: x[attribute_name] >=0.2, axis=1)]
        elif attribute_name == 'scale':
            data = data[data.apply(lambda x: x[attribute_name] >=500 or x[attribute_name] <=50, axis=1)]
        elif attribute_name == 'delta_scale':
            data = data[data.apply(lambda x: x[attribute_name] >=30, axis=1)]
        elif attribute_name == 'color_constancy_tran':
            data = data[data.apply(lambda x: x[attribute_name] <= 0.999, axis=1)]
        elif attribute_name == 'delta_color_constancy_tran':
            data = data[data.apply(lambda x: x[attribute_name] >= 0.00001, axis=1)]
        return data


    def _calc_metrics(self, boxes, anno, bound):
        """
        Calculate the evaluation metrics.
        """
        valid = ~np.any(np.isnan(anno), axis=1)
        if len(valid) == 0:
            print('Warning: no valid annotations')
            return None, None, None
        else:
            # calculate ious, dious and gious for success plot
            ious = iou(boxes[valid, :], anno[valid, :])
            dious = diou(boxes[valid, :], anno[valid, :])
            gious = giou(boxes[valid, :], anno[valid, :])
            # calculate center error for original precision plot
            center_errors = center_error(
                boxes[valid, :], anno[valid, :])
            # calculate normalized center error for the normalized precision plot
            norm_center_errors, flags = normalized_center_error(
                boxes[valid, :], anno[valid, :], bound)
            return ious, dious, gious, center_errors, norm_center_errors, flags


    def _calc_curves(self, ious, dious, gious, center_errors, norm_center_errors):
        """
        Calculate the evaluation curves.
        """
        ious = np.asarray(ious, float)[:, np.newaxis]
        dious = np.asarray(dious, float)[:, np.newaxis]
        gious = np.asarray(gious, float)[:, np.newaxis]
        center_errors = np.asarray(center_errors, float)[:, np.newaxis]
        norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

        thr_iou = np.linspace(0, 1, self.nbins_iou)[np.newaxis, :]
        thr_ce = np.arange(0, self.nbins_ce)[np.newaxis, :]
        thr_nce = np.linspace(0, 1, self.nbins_ce)[np.newaxis, :]

        bin_iou = np.greater(ious, thr_iou)
        bin_diou = np.greater(dious, thr_iou)
        bin_giou = np.greater(gious, thr_iou)
        bin_ce = np.less(center_errors, thr_ce)
        bin_nce = np.less(norm_center_errors, thr_nce)

        succ_curve = np.nanmean(bin_iou, axis=0)
        succ_dcurve = np.nanmean(bin_diou, axis=0)
        succ_gcurve = np.nanmean(bin_giou, axis=0)
        prec_curve = np.nanmean(bin_ce, axis=0)
        norm_prec_curve = np.nanmean(bin_nce, axis=0)

        return succ_curve, succ_dcurve, succ_gcurve, prec_curve, norm_prec_curve


    def report_robust(self, root_dir, tracker_names):
        """
        robust score under R-OPE
        """
        subset_report_dir = os.path.join(self.report_dir, self.subset)
        makedir(subset_report_dir)

        subset_analysis_dir = os.path.join(self.analysis_dir, self.subset)
        makedir(subset_analysis_dir)
        report_dir = os.path.join(subset_report_dir, tracker_names[0])
        makedir(report_dir)

        performance = {}
        for name in tracker_names:
            print('Evaluating Robust', name)

            single_report_file = os.path.join(subset_analysis_dir, '%s_%s_robust_%s.json'%(name, self.subset, str(self.repetition)))

            ious = {}
            times = {}
            center_errors = {}

            if os.path.exists(single_report_file):
                f = open(single_report_file,'r',encoding='utf-8')
                single_performance = json.load(f)            
                performance.update({name:single_performance})
                f.close()
                print('Existing result')
                continue
            else:
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})
            
            robust_list = []

            for s in range(len(self.dataset.seq_names)):
                num = self.dataset.seq_names[s]
        
                img_files, anno, restart_flag = self.dataset[s]
               
                print('Evaluate ' + ' ' + name + ' ' + num)
                seq_name = self.dataset.seq_names[s]

                corrcoef_path = os.path.join(self.root_dir, self.subset, num,'corrcoef_%s.txt'%num)
                corrcoef = pd.read_csv(corrcoef_path,header=None,names=['corrcoef'])
                corrcoef_mean = np.nanmean(corrcoef)

                restart_path = os.path.join(self.root_dir, self.subset, num,'restart_%s.txt'%num)
                restart = len(open(restart_path,'r').readlines()) if os.path.getsize(restart_path) != 0 else 1

                init_path = os.path.join(self.result_dir, name, self.subset, 'init_{}_{}_{}.txt'.format(name, num, self.repetition))
                init = len(open(init_path,'r').readlines()) if os.path.getsize(init_path) != 0 else 1

                robust_score = self.sigmoid(1/corrcoef_mean)*(1-init/restart)
                print(robust_score)

                performance[name]['seq_wise'].update({seq_name: {
                    'corrcoef_mean': corrcoef_mean,
                    'restart': restart,
                    'init': init,
                    'robust_score':robust_score}})
                
                robust_list.append(robust_score) 
            
            robust_overall = np.nanmean(np.array(robust_list))

            performance[name].update({'overall': {
            'robust_score':robust_overall}})
            
            with open(single_report_file, 'w') as f:
                json.dump(performance[name], f, indent=4)
                
        report_file = os.path.join(report_dir, 'robust.json')
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
                  

    def plot_curves_(self, report_files, tracker_names, attribute_name, rep):
        """
        Drow Plot
        """
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        report_dir = os.path.join(self.report_dir, self.subset,  tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))
        if rep == 'all':
            if attribute_name is not None:
                succ_file = os.path.join(report_dir, attribute_name+'_'+'success_plot_iou_all.png')
                succ_dfile = os.path.join(report_dir, attribute_name+'_'+'success_plot_diou_all.png')
                succ_gfile = os.path.join(report_dir, attribute_name+'_'+'success_plot_giou_all.png')
                prec_file = os.path.join(report_dir, attribute_name+'_'+'precision_plot_all.png')
                norm_prec_file = os.path.join(report_dir, attribute_name+'_'+'norm_precision_plot_all.png')
            else:
                succ_file = os.path.join(report_dir, 'overall_'+'success_plot_iou_all.png')
                succ_dfile = os.path.join(report_dir, 'overall_'+'success_plot_diou_all.png')
                succ_gfile = os.path.join(report_dir, 'overall_'+'success_plot_giou_all.png')
                prec_file = os.path.join(report_dir, 'overall_'+'precision_plot_all.png')
                norm_prec_file = os.path.join(report_dir, 'overall_'+'norm_precision_plot_all.png')
        else:
            if attribute_name is not None:
                succ_file = os.path.join(report_dir, attribute_name+'_'+'success_plot_iou_'+str(rep)+'.png')
                succ_dfile = os.path.join(report_dir, attribute_name+'_'+'success_plot_diou_'+str(rep)+'.png')
                succ_gfile = os.path.join(report_dir, attribute_name+'_'+'success_plot_giou_'+str(rep)+'.png')
                prec_file = os.path.join(report_dir, attribute_name+'_'+'precision_plot_'+str(rep)+'.png')
                norm_prec_file = os.path.join(report_dir, attribute_name+'_'+'norm_precision_plot_'+str(rep)+'.png')
            else:
                succ_file = os.path.join(report_dir, 'overall_'+'success_plot_iou_'+str(rep)+'.png')
                succ_dfile = os.path.join(report_dir, 'overall_'+'success_plot_diou_'+str(rep)+'.png')
                succ_gfile = os.path.join(report_dir, 'overall_'+'success_plot_giou_'+str(rep)+'.png')
                prec_file = os.path.join(report_dir, 'overall_'+'precision_plot_'+str(rep)+'.png')
                norm_prec_file = os.path.join(report_dir, 'overall_'+'norm_precision_plot_'+str(rep)+'.png')
        
        
        key = 'overall'

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by success score iou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_iou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_iou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_iou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on IoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
        
        # sort trackers by success score diou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_diou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_diou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_diou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on DIoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_dfile)
        fig.savefig(succ_dfile,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

          # sort trackers by success score giou
        tracker_names = list(performance.keys())
        succ = [t[key]['success_score_giou'] for t in performance.values()]
        inds = np.argsort(succ)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['success_curve_giou'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['success_score_giou']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower left', bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on VideoCube (based on GIoU)')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_gfile)
        fig.savefig(succ_gfile,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

        # sort trackers by precision score
        tracker_names = list(performance.keys())
        
        prec = [t[key]['precision_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot precision curves
        thr_ce = np.arange(0, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_ce,
                            performance[name][key]['precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            

            legends.append('%s: [%.3f]' % (name, performance[name][key]['precision_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Location error threshold',
               ylabel='Precision',
               xlim=(0, thr_ce.max()), ylim=(0, 1),
               title='Precision plots on VideoCube')
        ax.grid(True)
        fig.tight_layout()

        print('Saving precision plots to', prec_file)
        fig.savefig(prec_file, dpi=300)

        # plot normalized precision curves
        tracker_names = list(performance.keys())
        # prec = [t[key]['normalized_precision_score'] for t in performance.values()]
        prec = [t[key]['norm_prec_score'] for t in performance.values()]

        inds = np.argsort(prec)[::-1]

        tracker_names = [tracker_names[i] for i in inds]

        # plot normalized precision curves
        thr_nce = np.linspace(0, 1, self.nbins_ce)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_nce,
                            performance[name][key]['normalized_precision_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (name, performance[name][key]['norm_prec_score']))
        matplotlib.rcParams.update({'font.size': 7.4})
        # legend = ax.legend(lines, legends, loc='center left', bbox_to_anchor=(1, 0.5))
        legend = ax.legend(lines, legends, loc='lower right', bbox_to_anchor=(1., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Normalized location error threshold',
               ylabel='Normalized precision',
               xlim=(0, thr_nce.max()), ylim=(0, 1),
               title='Normalized precision plots on VideoCube')
        ax.grid(True)
        fig.tight_layout()

        print('Saving normalized precision plots to', norm_prec_file)
        fig.savefig(norm_prec_file, dpi=300)
    

    def _record(self, record_file, time_file, boxes, times):
        np.savetxt(record_file, boxes, fmt='%d', delimiter=',')
        print('Results recorded at', record_file)

        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')


    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))


    def eye_report(self, root_dir, tracker_names):
        """
        Calculate in eye tracking subset
        """
        attribute_name='normal'
        report_dir = os.path.join(self.report_dir, 'eye')
        performance = self.report(tracker_names, attribute_name)
