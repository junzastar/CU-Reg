# [MICCAI'24] Epicardium Prompt-guided Real-time Cardiac Ultrasound Frame-to-volume Registration
This is the PyTorch implementation of MICCAI'24 paper.

<p align="center">
<img src="figures/pineline.pdf" alt="intro" width="100%"/>
</p>

## Abstract
> A comprehensive guidance view for cardiac interventional surgery can be provided by the real-time fusion of the intraoperative 2D images and preoperative 3D volume based on the ultrasound frame-to-volume registration. 
However, cardiac ultrasound images are characterized by a low signal-to-noise ratio and small differences between adjacent frames, coupled with significant dimension variations between 2D frames and 3D volumes to be registered, resulting in real-time and accurate cardiac ultrasound frame-to-volume registration being a very challenging task. 
This paper introduces a lightweight end-to-end Cardiac Ultrasound frame-to-volume Registration network, termed CU-Reg.
Specifically, the proposed model leverages epicardium prompt-guided anatomical clues to reinforce the interaction of 2D sparse and 3D dense features, followed by a voxel-wise local-global aggregation of enhanced features, thereby boosting the cross-dimensional matching effectiveness of low-quality ultrasound modalities. 
We further embed an inter-frame discriminative regularization term within the hybrid supervised learning to increase the distinction between adjacent slices in the same ultrasound volume to ensure registration stability. 
Experimental results on the reprocessed CAMUS dataset demonstrate that our CU-Reg surpasses existing methods in terms of registration accuracy and efficiency, meeting the guidance requirements of clinical cardiac interventional surgery. 

## Installation
**Conda virtual environment**

We recommend using conda to setup the environment.

If you have already installed conda, please use the following commands.

```bash
conda create -n CU-Reg python=3.8
conda activate CU-Reg
pip install -r requirements.txt
```

## Dataset
Download the processed CAMUS dataset, you can download it [here](https://github.com/junzastar/CU-Reg.git).

## Evaluation
Please download our trained model [here](https://github.com/junzastar/CU-Reg.git) and put it in the 'CU_Reg/pre_trained/trained_models' directory. Then, you can have a quick evaluation using the following command.
```bash
python test.py
```

## Train
In order to train the model, remember to download the complete dataset.

train.py is the main file for training. You can simply start training using the following command.
```bash
python train.py
```

## Citation
If you find the code useful, please cite our paper.
```latex

```
Any questions, please feel free to contact
Long Lei (longlei@cuhk.edu.hk), \
Jun Zhou (zachary-jun.zhou@connect.polyu.hk), \
Jialun Pei (jialunpei@cuhk.edu.hk).

## Acknowledgment
Our code is developed based on [FVR-Net](https://github.com/DIAL-RPI/FVR-Net.git).
