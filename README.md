# VideoCube Python Toolkit

> UPDATE:<br>
> [2023.02.08] To make it easier for users to research with VideoCube, we have selected 150 representative sequences from the original version (500 sequences) to form ***VideoCube-Tiny***. This toolkit has been updated to support the VideoCube-Tiny, and you can find a demo to select *tiny* or *full* version in [test.py](./test.py). <br>
> [2022.06.20] Update the [SiamFC](https://github.com/huuuuusy/videocube-toolkit/blob/master/tracker/siamfc.py) based on [issue #2](https://github.com/huuuuusy/videocube-toolkit/issues/2) and [issue #3](https://github.com/huuuuusy/videocube-toolkit/issues/3). <br>
> [2022.04.25] Update the [time](https://github.com/huuuuusy/videocube-toolkit/blob/master/videocube/trackers/__init__.py) calculation method based on [issue #1](https://github.com/huuuuusy/videocube-toolkit/issues/1). <br>
> [2022.03.03] Update the toolkit installation, dataset download instructions and a concise example. Now the basic function of this toolkit has been finished. <br>

This repository contains the official python toolkit for running experiments and evaluate performance on [VideoCube](http://videocube.aitestunion.com/) benchmark. The code is written in pure python and is compile-free.

[VideoCube](http://videocube.aitestunion.com/) is a high-quality and large-scale benchmark to create a challenging real-world experimental environment for Global Instance Tracking (GIT) task. If you use the VideoCube database or toolkits for a research publication, please consider citing:

```Bibtex
@ARTICLE{9720246,
  author={Hu, Shiyu and Zhao, Xin and Huang, Lianghua and Huang, Kaiqi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Global Instance Tracking: Locating Target More Like Humans}, 
  year={2023},
  volume={45},
  number={1},
  pages={576-592},
  doi={10.1109/TPAMI.2022.3153312}}
```

&emsp;\[[Project](http://videocube.aitestunion.com/)\]\[[PDF](https://arxiv.org/abs/2202.13073)\]


![](./demo/demo.gif)

## Table of Contents

- [VideoCube Python Toolkit](#videocube-python-toolkit)
  - [Table of Contents](#table-of-contents)
    - [Toolkit Installation](#toolkit-installation)
    - [Dataset Download](#dataset-download)
    - [A Concise Example](#a-concise-example)
      - [How to Define a Tracker?](#how-to-define-a-tracker)
      - [How to Run Experiments on VideoCube?](#how-to-run-experiments-on-videocube)
      - [How to Evaluate Performance?](#how-to-evaluate-performance)
    - [Issues](#issues)
    - [Contributors](#contributors)

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

- Note that we have released a tiny version named VideoCube-Tiny. The original full version includes ***500 sequences (1.4T)***, while the tiny version includes ***150 sequences (344G)***.

- Check the number of files in each subset and run the unzipping script. Before unzipping:

  - the *train* subset of *full version* should includ 456 files (455 data files and an unzip_train bash); the *train* subset of *tiny version* should includ 129 files (128 data files and an unzip_train bash)

  - the *val* subset of *full version* should includ 69 files (68 data files and an unzip_val bash); the *val* subset of *tiny version* should includ 22 files (21 data files and an unzip_val bash)

  - the *test* subset of *full version* should includ 140 files (139 data files and an unzip_test bash); the *test* subset of *tiny version* should includ 41 files (40 data files and an unzip_test bash)

- Run the unzipping script in each subset folder, and delete the script after decompression.

- Taking *val* subset of *full version* as an example, the folder structure should follow:

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

- Unzip attribute.zip in info data. Attention that we only provide properties files for *train* and *val* subsets. For ground-truth files, we only offer a small number of annotations (restart frames) for sequences that belong to the *test* subset. Please upload the final results to the server for evaluation.

- Rename and organize folders as follows (this example is illustrated for *full version*, while *tiny version* has the similar data structure)：

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

[test.py](./test.py) is a simple example on how to use the toolkit to define a tracker, run experiments on VideoCube and evaluate performance.

#### How to Define a Tracker?

To define a tracker using the toolkit, simply inherit and override `init` and `update` methods from the [`Tracker`](./videocube/trackers/__init__.py) class. You can find an example in this [page](./tracker/siamfc.py). Here is a simple example:

```Python
from videocube.trackers import Tracker

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
        )
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box
```

#### How to Run Experiments on VideoCube?

Instantiate an [`ExperimentVideoCube`](./videocube/experiments/videocube.py) object, and leave all experiment pipelines to its `run` method:

```Python
from videocube.experiments import ExperimentVideoCube

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()

# setup experiment (validation subset)
experiment = ExperimentVideoCube(
  root_dir='SOT/VideoCube', # VideoCube's root directory
  save_dir= os.path.join(root_dir,'result'), # the path to save the experiment result
  subset='val', # 'train' | 'val' | 'test'
  repetition=1,
  version='tiny' # set the version as 'tiny' or 'full' 
)
experiment.run(
  tracker, 
  visualize=False,
  save_img=False, 
  method='restart' # method in run function means the evaluation mechanism, you can select the original mode (set none) or the restart mode (set restart)
  )
```

#### How to Evaluate Performance?

For evaluation in OPE mechanism, please use the `report` method of `ExperimentVideoCube` for this purpose:

```Python
# ... run experiments on VideoCube ...

# report tracking performance
experiment.report([tracker.name],attribute_name)
```

For evaluation in R-OPE mechanism, please use the `report` and `report_robust` method of `ExperimentVideoCube` for this purpose:

```Python
# ... run experiments on VideoCube ...

# report tracking performance
experiment.report([tracker.name],attribute_name)
experiment.report_robust([tracker.name])
```

Attention, when evaluated on the __test subset__, you will have to submit your results to the [evaluation server](http://videocube.aitestunion.com/submit_supports) for evaluation. The `report` function will generate a `.zip` file which can be directly uploaded for submission. For more instructions, see [submission instruction](http://videocube.aitestunion.com/submit_supports).

See public evaluation results on [VideoCube's leaderboard (OPE Mechanism)](http://videocube.aitestunion.com/leaderboard) and [VideoCube's leaderboard (R-OPE Mechanism)](http://videocube.aitestunion.com/leaderboard_restart).

### Issues

Please report any problems or suggessions in the [Issues](https://github.com/huuuuusy/videocube-toolkit/issues) page.

### Contributors
- [Shiyu Hu](https://github.com/huuuuusy)
- [Lianghua Huang](https://github.com/huanglianghua)
