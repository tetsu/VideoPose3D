## Installation Instruction for Cuda 10.0


1. Install Anaconda, and create conda environment for detectron
   ```
   conda create -n detectron python=2.7
   conda activate detectron
   ```

1. Install pytorch with anaconda

   For Cuda 10.0
   ```
   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   ```
   For Cuda 10.1
   ```
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   ```

1. Install caffe2 with anaconda
   ```
   conda install -c caffe2 caffe2
   ```

1. Install other dependencies
   ```
   conda install mkl mkl-include matplotlib mock scipy cmake cffi typing Cython
   ```

   Or
   ```
   pip install numpy>=1.13 pyyaml>=3.12 matplotlib opencv-python>=3.2 Cython mock scipy mkl mkl-include setuptools cmake cffi typing
   ```


1. Install Coco API
   ```
   cd ~/github (or any directory of your choice)
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi/PythonAPI
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

1. Test Detectron
   ```
   python detectron/tests/test_spatial_narrow_as_op.py
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

1. Create conda environment for VideoPose3D
   ```
   conda create -n VideoPose3D python=3.6
   conda activate VideoPose3D
   ```

1. Install pytorch with anaconda

   For Cuda 10.0
   ```
   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   ```
   For Cuda 10.1
   ```
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   ```

1. Copy config file from Detectron to VideoPose3D
   ```
   cp ~/github/Detectron/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml ~/github/VideoPose3D/detectron_tools
   ```

1. Download and setup pretrained h36m file
   ```
   cd data
   wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
   python prepare_data_h36m.py --from-archive h36m.zip
   cd ..
   ```

1. Download [Weight File](https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl) for e2e_keypoint_rcnn_R-101-FPN_s1x.yaml, and move it to detectron_tools directory. More files [here](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)
   ```
   mv ~/Download/model_final.pkl ~/github/VideoPose3D/detectron_tools
   ```

## Download or Prepare for a video file, and split into frames
1. To download a video file from Youtube, instal youtube-dl. For example, you can install it through pip.
   ```
   pip install youtube-dl
   ```

1. Download a video file from youtube. One example below:
   ```
   youtube-dl -f mp4 https://www.youtube.com/watch?v=LXO-jKksQkM
   ```

1. Install ffmpeg via apt (ffmpeg installed by pip does not work properly)
   ```
   sudo apt install ffmpeg
   ```

1. Split the video file into images using ffmpeg. (change file name and folder as you want, but make sure the folder exists.)
   ```
   ffmpeg -i input_video_file.mp4 -r 25 splitted_scating/InputVideoFileFolder/output%04d.png
   ```

## Run Detectron

1. Go to detectron_tools directory, and run detectron
   ```
   cd ~/github/VideoPose3D/detectron_tools
   python infer_simple.py --cfg e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir output --wts model_final.pkl splitted_scating
   ```
   Example
   ```
   python infer_simple.py --cfg e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir output/pumped --output-ext png --image-ext jpg --wts model_final.pkl video/pumped
   ```


## Setup VideoPose3D Environment

1. Create a new Anaconda environment for VideoPose3D
   ```
   conda create -n VideoPose3D python=3.6 numpy=1.16.2 matplotlib scipy yaml pyyaml=3.12 protobuf opencv cython future
   conda activate VideoPose3D
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   conda install cudnn
   ```

1. Install Coco API in VideoPose3D environment
   ```
   cd ~/github (or any directory of your choice)
   git clone https://github.com/cocodataset/cocoapi.git
   cd cocoapi/PythonAPI
   make install
   ```

1. Install Detectron in VideoPose3D environment
   ```
   cd ~/github (or any directory of your choice)
   git clone git@github.com:facebookresearch/Detectron.git
   cd Detectron
   pip install -r requirements.txt
   make
   ```

1. Download and locate necessary files
   ```
   cd ~/github/VideoPose3D/data
   wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
   python prepare_data_h36m.py --from-archive h36m.zip

   cd ~/github/VideoPose3D/checkpoint
   wget https://dl.fbaipublicfiles.com/video-pose-3d/d-pt-243.bin
   ```

1. Move the video file to project's root directory (example in this case)
   ```
   mv ~/github/VideoPose3D/detection_tools/input_video_file.mp4 ~/github/VideoPose3D/
   ```

1. Run VideoPose3D
   ```
   python run_wild.py -k detections -arc 3,3,3,3,3 -c checkpoint --evaluate d-pt-243.bin --render --viz-subject S1 --viz-action Directions --viz-video InTheWildData/out_cutted.mp4 --viz-camera 0 --viz-output output_scater.mp4 --viz-size 5 --viz-downsample 1 --viz-skip 9
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