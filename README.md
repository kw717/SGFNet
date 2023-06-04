# SGFNet

This project provides the code and results for 'SGFNet: Semantic-Guided Fusion Network for
RGB-Thermal Semantic Segmentation
', IEEE TCSVT, 2023. [IEEE link](https://ieeexplore.ieee.org/abstract/document/10138593)

### Requirements
  python 3.7/3.8 + pytorch 1.9.0 (built on [EGFNet](https://github.com/ShaohuaDong2021/EGFNet))
  
### Evaluation

+ Download the following pre-trained model.
+ [SGFNet.pth](https://pan.baidu.com/s/1tdIY6ZgZHSpEFW2hoDPD8w)
 (code: 466s)  
+ [SGFNet_pst.pth](https://pan.baidu.com/s/1zaBHx3V1tm5JqqpMVr9Q4g) (code: lpte)
+ test on MFNet dataset:
  ```shell
  python test.py --logdir [run logdir] --data-root [path to dataset] --pth [path to pre-trained model]
  ```
+ test on PST900 dataset:
  ```shell
  python test_pst.py --logdir [run logdir] --data-root [path to dataset] --pth [path to pre-trained model]
  ```
### Segmentation maps
   We provide segmentation maps on MFNet dataset(SGFNet_MF.zip) and PST900 dataset(SGFNet_pst.zip).
### Citation
       @ARTICLE{Wang_2023_SGFNet,
                author={Wang, Yike and Li, Gongyang and Liu, Zhi},
                journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
                title={SGFNet: Semantic-Guided Fusion Network for RGB-Thermal Semantic Segmentation}, 
                year={2023},
                doi={10.1109/TCSVT.2023.3281419}}

                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at kevinwangyike@qq.com or shuwangyike@shu.edu.cn.
