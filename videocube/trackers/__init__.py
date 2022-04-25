from __future__ import absolute_import

from typing import Union
import torch

import numpy as np
import time

import cv2 as cv

from ..utils.metrics import iou
import concurrent.futures

class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic
        if self.is_using_cuda:
            print('Detect the CUDA devide')
            self._timer_start = torch.cuda.Event(enable_timing=True)
            self._timer_stop = torch.cuda.Event(enable_timing=True)
        self._timestamp = None
    
    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()

    @property
    def is_using_cuda(self):
        self.cuda_num = torch.cuda.device_count()
        if self.cuda_num == 0:
            return False
        else:
            return True

    def _start_timing(self) -> Union[float, None]:
        if self.is_using_cuda:
            self._timer_start.record()
            timestamp = None
        else:
            timestamp = time.time()
            self._timestamp = timestamp
        return timestamp

    def _stop_timing(self) -> float:
        if self.is_using_cuda:
            self._timer_stop.record()
            torch.cuda.synchronize()
            # cuda event record return duration in milliseconds.
            duration = self._timer_start.elapsed_time(
                self._timer_stop
            )
            duration /= 1000.0
        else:
            duration = time.time() - self._timestamp
        return duration

    def track(self,seq_name, img_files, anno, restart_flag, visualize, seq_result_dir, save_img, method):
        frame_num = len(img_files)
        box = anno[0,:] # the information of the first frame 
        boxes = np.zeros((frame_num, 4)) # save the tracking result
        boxes[0] = box 
        times = np.zeros(frame_num) # save time

        fail_count = 0 # fail_count records the failures in R-OPE mechanism

        init_positions = [] # save the restart locations
        if visualize:
            display_name = 'Display: ' + seq_name
            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)
        
        with concurrent.futures.ProcessPoolExecutor() as executor: 
            executor.map(cv.imread, img_files)

        for f, img_file in enumerate(img_files):

            image = cv.imread(img_file)
            height = image.shape[0]
            width = image.shape[1]
            img_resolution = (width,height)
                
            # start_time = time.time() 
            self._start_timing()
            if f == 0: 
                self.init(image, box)
                times[f] = self._stop_timing()
            if fail_count >= 10 and method == 'restart' and f in restart_flag:
                # the tracker will be restarted when the cumulative number of failures reaches 10
                print('init again in %s' % f)                
                init_positions.append(f)
                self.init(image, anno[f,:])
                fail_count = 0
            else:
                frame_box = self.update(image) 
                frame_box = np.rint(frame_box)
                times[f] = self._stop_timing()

                current_gt = anno[f,:].reshape((1,4))
                frame_box = np.array(frame_box)
                track_result = frame_box.reshape((1,4))
                bound = img_resolution
                seq_iou = iou(current_gt, track_result, bound=bound)
                
                # check failures
                if method == 'restart' and (anno[f,:] != np.array([0,0,0,0])).all(): 
                    if seq_iou < 0.5: 
                        # failure occures in present frame
                        fail_count += 1
                    else: 
                        # re-locate the target
                        fail_count = 0
                        
                boxes[f, :] = frame_box

                if method == 'restart':
                    print(seq_name, self.name,' Tracking %d/%d' % (f, frame_num-1), 'time:%.2f' % times[f], 'fail count:', fail_count, frame_box)
                else:
                    print(seq_name, self.name,' Tracking %d/%d' % (f, frame_num-1), 'time:%.2f' % times[f], frame_box)
                
                if save_img or visualize:
                    frame_disp = image.copy()
                    state = [int(s) for s in frame_box]
                    state[0] = 0 if state[0] < 0 else state[0]
                    state[1] = 0 if state[1] < 0 else state[1]
                    state[2] = width-state[0] if state[0]+state[2] > width else state[2]
                    state[3] = height-state[1] if state[1]+state[3] > height else state[3] 
                    font_face = cv.FONT_HERSHEY_SIMPLEX 
                    cv.putText(frame_disp,'No.%06d'%(f), (50, 100), font_face, 0.8, (0, 255, 0), 2)
                    if (anno[f,:] != np.array([0,0,0,0])).all():
                        cv.putText(frame_disp,'seq iou: %2f'%(seq_iou), (50, 130), font_face, 0.8, (0, 255, 0), 2)

                    cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),(0, 255, 0), 5)
                    gt = [int(s) for s in anno[f,:]]
                    cv.rectangle(frame_disp, (gt[0], gt[1]), (gt[2] + gt[0], gt[3] + gt[1]),(0, 0, 255), 5)

                if visualize:
                    cv.imshow(display_name, frame_disp)
                if save_img:
                    save_path = "{}/{:>06d}.jpg".format(seq_result_dir, f)
                    cv.imwrite(save_path, frame_disp)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
          
        if visualize:
            cv.destroyAllWindows()
        
        if method == None:
            return boxes, times
        elif method == 'restart':
            return boxes, times, init_positions
    
