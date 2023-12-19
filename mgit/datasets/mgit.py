from __future__ import absolute_import, print_function

import glob
import json
import numpy as np
import os
import pandas as pd
import six


class MGIT(object):
    r"""`MGIT <http://videocube.aitestunion.com>`_ Dataset.

    Publication:
        ``A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship``, S. Hu, D. Zhang, M. Wu, X. Feng, X. Li, X. Zhao, K. Huang
        Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. 2023
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        split (string, optional): Specify ``train``, ``val`` or ``test``
            subset of MGIT.
    """

    def __init__(self, root_dir, split, version='full'):
        super(MGIT, self).__init__()
        assert split in ['train', 'val', 'test'], 'Unknown subset.'
        self.base_path = root_dir
        self.split = split
        self.version = version  # temporarily, the toolkit only support tiny version of MGIT

        f = open(os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mgit.json'), 'r', encoding='utf-8')
        self.infos = json.load(f)[self.version]
        f.close()

        self.sequence_list = self.infos[self.split]

        if split in ['train', 'val', 'test']:
            self.seq_dirs = [os.path.join(root_dir, 'data', split, s, 'frame_{}'.format(s)) for s in self.sequence_list]
            self.anno_files = [os.path.join(root_dir, 'attribute', 'groundtruth', '{}.txt'.format(s)) for s in
                               self.sequence_list]
            self.restart_files = [os.path.join(root_dir, 'attribute', 'restart', '{}.txt'.format(s)) for s in
                                  self.sequence_list]

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
            if not index in self.sequence_list:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.sequence_list.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))

        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        nlp_path = './mgit/datasets/mgit_nlp/{}.xlsx'.format(
            self.sequence_list[index])
        nlp_tab = pd.read_excel(nlp_path)
        nlp_rect = nlp_tab.iloc[:, [14]].values
        nlp_rect = nlp_rect[-1, 0]

        restart_flag = np.loadtxt(self.restart_files[index], delimiter=',', dtype=int)

        return img_files, anno, nlp_rect, restart_flag

    def __len__(self):
        return len(self.sequence_list)
