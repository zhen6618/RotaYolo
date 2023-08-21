# RotaYolo
In the task of detecting oriented bounding boxes (OBBs), orientation regression-based OBB representations and point-based OBB representations suffer from the loss discontinuity problem at some extreme orientations and aspect ratios, that is, tiny changes of the OBB orientations and aspect ratios will result in large model loss jumps in these extreme cases. Moreover, existing methods using convolutional neural networks (CNNs) can only extract orientation features under local receptive fields. To address the above problems, we propose Eigen Loss and Angle Convolution in this paper. By mapping OBBs into eigenmatrices and center vectors, Eigen Loss converts the prediction of OBBs into the prediction of eigenmatrices and center vectors, achieving continuity at arbitrary orientations and aspect ratios. Meanwhile, by adaptively rotating grouped feature maps of CNNs and applying a lightweight convolutional network to the rotated feature maps, Angle Convolution captures global information among feature maps of different groups, achieving local and global orientation feature extraction with a lightweight structure and simplified implementation. Furthermore, to evaluate the effectiveness of Eigen Loss and Angle Convolution, we propose an oriented object detector called RotaYolo which applies them to the improved Yolov7 detector. Experiments on DOTA-v1.5 and HRSC2016 datasets demonstrate that RotaYolo outperforms current state-of-the-art (SOTA) methods. Moreover, it is verified that some SOTA methods applying Eigen Loss and Angle Convolution will further improve detection accuracy while maintaining competitive detection speed. 


![Results of Angle Convolution](步骤2复制的链接)

![Results on DOTA dataset]("https://github.com/zhen6618/RotaYolo/blob/main/DOTA.png")

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

