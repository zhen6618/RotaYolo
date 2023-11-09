# RotaYolo
Due to the frequent variability of object orientation, accurate prediction of orientation information remains a challenge in oriented object detection. To better extract orientation-related information, current methods primarily focus on the design of reasonable representations of oriented bounding boxes (OBBs) and rotation-sensitive feature extraction. However, existing OBB representations often suffer from boundary discontinuity and representation ambiguity problems. Methods of designing continuous and unambiguous regression losses do not essentially solve such problems. Gaussian bounding boxes (GBBs) avoid these OBB representation problems, but directly regressing GBBs is susceptible to numerical instability. In this paper, we propose linear GBB (LGBB), a novel OBB representation. By linearly transforming the elements of GBB, LGBB does not have the boundary discontinuity and representation ambiguity problems, and have high numerical stability. On the other hand, current rotation-sensitive feature extraction methods based on convolutions can only extract features under a local receptive field, which is slow in aggregating rotation-sensitive features. To address this issue, we propose ring-shaped rotated convolution (RRC). By adaptively rotating feature maps to arbitrary orientations, RRC extracts rotation-sensitive features under a ring-shaped receptive field, rapidly aggregating rotation-sensitive features and contextual information. RRC can be applied to various models in a plug-and-play manner. Experimental results demonstrate that the proposed LGBB and RRC are effective and achieve state-of-the-art (SOTA) performance. By integrating LGBB and RRC into various models, the detection accuracy is effectively improved on DOTA and HRSC2016 datasets.

<div align=center>
<img src="https://github.com/zhen6618/RotaYolo/blob/main/Conv.png" width="500px">

<img src="https://github.com/zhen6618/RotaYolo/blob/main/DOTA.png" width="800px">
</div>


# Training
```
# Single GPU training
python train.py --workers 8 --device 0 --batch-size 2 --data data/dota.yaml --img 1024 1024 --cfg cfg/training/RotaYolo_RotaConv.yaml --weights '' --hyp data/hyp.scratch.dota.yaml

# Multiple GPU training
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 8 --data data/dota.yaml --img 1024 1024 --cfg cfg/training/RotaYolo_RotaConv.yaml --weights '' --hyp data/hyp.scratch.dota.yaml
```

# Detecting
```
python detect.py --weights 'weights/best.pt' --source 'datasets/DOTA/demo.png' --img-size 1024 --conf-thres 0.5 --iou-thres 0.2 --device 0
```

# Citation

