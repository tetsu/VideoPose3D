# Installation Instruction for Cuda 10.0


1. Install Anaconda, and create conda environment for detectron
   ```
   conda create -n detectron python=2.7
   conda activate detectron
   ```

1. Install pytorch with anaconda
   ```
   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   ```

1. Install caffe2 with anaconda
   ```
   conda install -c caffe2 caffe2
   ```

1. Install other dependencies
   ```
   pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 Cython mock scipy mkl mkl-include setuptools cmake cffi typing
   ```

1. Install Coco API
   ```
   cd ~/github (or any directory of your choice)
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi
   make install
   ```

1. Install Detectron
   ```
   cd ~/github (or any directory of your choice)
   git clone git@github.com:facebookresearch/Detectron.git
   cd Detectron
   pip install -r requirements.txt
   make
   ```

1. Install ffmpeg to your system (Ubuntu in this case)
   ```
   sudo apt update
   sudo apt install ffmpeg
   ```

1. Clone VideoPose3D, go to detectron_tools directory
   ```
   cd ~/github (or any directory of your choice)
   git clone git@github.com:tobiascz/VideoPose3D.git
   ```

1. Copy config file from Detectron
   ```
   cp ~/github/Detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml ~/github/VideoPose3D/detectron_tools
   ```

1. Download Weight File: [File](https://www.dropbox.com/sh/vi5byf0du9g50lw/AABGVezeuHuipOzaFGdGGvbaa?dl=0), and move it to detectron_tools directory
   ```
   mv ~/Download/model_final.pkl ~/github/VideoPose3D/detectron_tools
   ```

1. Go to detectron_tools directory, and run detectron
   ```
   cd ~/github/VideoPose3D/detectron_tools
   python infer_simple.py --cfg e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir output --wts model_final.pkl splitted_scating
   ```

1. Run 3D
   ```
   python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video "out_cutted.mp4" --viz-camera 0 --viz-output output_video.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 0

   python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video "out_cutted.mp4" --viz-camera 0 --viz-output output.gif --viz-size 5 --viz-downsample 1

   python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video "out_cutted.mp4" --viz-camera 0 --viz-output output.mp4 --viz-size 5 --viz-downsample 1

   python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video InTheWildData/out_cutted.mp4 --viz-camera 0 --viz-output output_scater.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 9
   ```

## Detectron Installation by building

1. Install prerequisits for building pytorch 2
   ```
   pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 Cython mock scipy mkl mkl-include setuptools cmake cffi typing
   ```

1. Clone pytorch git repo, move to pytorch directory, and update submodule
   ```
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch
   git submodule update --init --recursive
   cd ..
   ```

1. Instal pytorch and caffe2
   ```
   FULL_CAFFE2=1 python setup.py install
   ```


## References
1. [Detectronをうごかしてみた](https://qiita.com/1O1/items/d3b982a76b1c43401acb)
1. [cuda10+cuDNN7.3でpytorch+caffe2をビルド](https://eigo.rumisunheart.com/2018/09/26/installing-pytorch-and-caffe2-on-cuda10-and-cudnn7/)
1. [Error while installing pytorch #16602](https://github.com/pytorch/pytorch/issues/16602)
1. [VideoPose3Dを動かしてみた（Detectronから）](https://qiita.com/timtoronto634/items/ee018ac89e6b9f779194)
1. [Detectron](https://github.com/facebookresearch/Detectron)
1. [Installing Detectron](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)
1. [Detectron2](https://github.com/facebookresearch/detectron2)
1. [VideoPose3D](https://github.com/tetsu/VideoPose3D)
1. [Download the weights file with the coco keypoints #2](https://github.com/tobiascz/VideoPose3D/issues/2)
1. [Caffe2 installation with conda](https://anaconda.org/caffe2/caffe2)
1. [Pytorch official installation with conda](https://pytorch.org/get-started/locally/)