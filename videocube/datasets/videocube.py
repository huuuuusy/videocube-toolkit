from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six


class VideoCube(object):
    r"""`VideoCube <http://videocube.aitestunion.com>`_ Dataset.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of VideoCube.
    """
    def __init__(self, root_dir, subset):
        super(VideoCube, self).__init__()
        self.root_dir = root_dir
        self.subset = subset
        
        self.seq_names = ['%03d'%num for num in np.loadtxt(os.path.join(root_dir, 'data', '{}_list.txt'.format(subset)))]
        
        if subset in ['train','val','test']:
            self.seq_dirs = [os.path.join(root_dir, 'data', subset, s,'frame_{}'.format(s)) for s in self.seq_names]
            self.anno_files = [os.path.join(root_dir,'attribute', 'groundtruth','{}.txt'.format(s)) for s in self.seq_names]
            self.restart_files = [os.path.join(root_dir,'attribute', 'restart','{}.txt'.format(s)) for s in self.seq_names]

        elif subset == 'eye':
            self.seq_dirs = [os.path.join(root_dir, 'data', 'test', s,'frame_{}'.format(s)) for s in self.seq_names]
            self.anno_files = [os.path.join(root_dir,'attribute', 'groundtruth','{}.txt'.format(s)) for s in self.seq_names]
            self.restart_files = [os.path.join(root_dir,'attribute', 'restart','{}.txt'.format(s)) for s in self.seq_names]
    
    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple:
                (img_files, anno, restart_flag), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``restart_flag`` is a list of
                restart frames.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, restart_flag
        

    def __len__(self):
        return len(self.seq_names)

