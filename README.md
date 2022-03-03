# VideoCube Python Toolkit

> UPDATE:<br>
> [2022.03.03] Update the toolkit installation and dataset download instructions.<br>

This repository contains the official python toolkit for running experiments and evaluate performance on [VideoCube](http://videocube.aitestunion.com/) benchmark. The code is written in pure python and is compile-free.

[VideoCube](http://videocube.aitestunion.com/) is a high-quality and large-scale benchmark to create a challenging real-world experimental environment for Global Instance Tracking (GIT) task. If you use the VideoCube database or toolkits for a research publication, please consider citing:

```Bibtex
@ARTICLE{9720246,
author={Hu, Shiyu and Zhao, Xin and Huang, Lianghua and Huang, Kaiqi},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Global Instance Tracking: Locating Target More Like Humans}, 
year={2022},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2022.3153312}}
```

&emsp;\[[Project](http://videocube.aitestunion.com/)\]\[[PDF](https://arxiv.org/abs/2202.13073)\]


![](./demo/demo.gif)

## Table of Contents

* [Toolkit Installation](#toolkit-installation)
* [Dataset Download](#dataset-download)
* [A Concise Example](#a-concise-example) (Coming soon)
* [Issues](#issues)
* [Contributors](#contributors)

### Toolkit Installation

Clone the repository and install dependencies:

```
git clone https://github.com/huuuuusy/videocube-toolkit.git
pip install -r requirements.txt
```

Then directly copy the `videocube` folder to your workspace to use it.

### Dataset Download

Please view the [Download](http://videocube.aitestunion.com/downloads) page in the project website.

The VideoCube dataset includes 500 sequences, divided into three subsets (*train*/*val*/*test*). The content distribution in each subset still follows the [6D principle](http://videocube.aitestunion.com/videocube) proposed in the [GIT paper](https://arxiv.org/abs/2202.13073).

The dataset download and file organization process is as follows：

- Download three subsets ([*train*](http://videocube.aitestunion.com/downloads_dataset/train_data)/[*val*](http://videocube.aitestunion.com/downloads_dataset/val_data)/[*test*](http://videocube.aitestunion.com/downloads_dataset/test_data)) and the [*info*](http://videocube.aitestunion.com/downloads_dataset/info) data via [Download](http://videocube.aitestunion.com/downloads) page in the project website.

- Check the number of files in each subset and run the unzipping script. Before unzipping:

  - the *train* subset should includ 456 files (455 data files and an unzip_train bash)

  - the *val* subset should includ 69 files (68 data files and an unzip_val bash)

  - the *test* subset should includ 140 files (139 data files and an unzip_test bash)

- Run the unzipping script in each subset folder, and delete the script after decompression.

- Taking *val* subset as an example, the folder structure should follow:

```
|-- val/
|  |-- 005/
|  |  |-- frame_005/
|  |  |  |-- 000000.jpg/
|  |  |      ......
|  |  |  |-- 016891.jpg/
|  |-- 008/
|  |   ......
|  |-- 486/
|  |-- 493/
```

- Unzip attribute.zip in info data.

- Rename and organize folders as follows：

```
|-- VideoCube/
|  |-- data/
|  |  |-- train/
|  |  |  |-- 002/
|  |  |  |   ......
|  |  |  |-- 499/
|  |  |-- val/
|  |  |  |-- 005/
|  |  |  |   ......
|  |  |  |-- 493/
|  |  |-- test/
|  |  |  |-- 001/
|  |  |  |   ......
|  |  |  |-- 500/
|  |  |-- train_list.txt
|  |  |-- val_list.txt
|  |  |-- test_list.txt
|  |-- attribute/
|  |  |-- absent/
|  |  |-- color_constancy_tran/
|  |  |   ......
|  |  |-- shotcut/
```

### A Concise Example

test.py is a simple example on how to use the toolkit to define a tracker, run experiments on VideoCube and evaluate performance.

The more detailed introduction will be uploaded soon.

### Issues

Please report any problems or suggessions in the [Issues](https://github.com/huuuuusy/videocube-toolkit/issues) page.

### Contributors
- [Shiyu Hu](https://github.com/huuuuusy)
- [Lianghua Huang](https://github.com/huanglianghua)
