# RetinaFace in PyTorch

只需要face detection的功能，所以移除了Retinaface中特征点的分支，并进行灰度图输入训练，输入尺寸为128 \* 128。
``` Shell
python train.py
```

10.26
已经修改使用 resnet18，尝试缩小主干网络，减少模型耗时的同时保证模型精度。

resnet网络修改
```Shell
...
        self.conv1_r = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
...
```
11.11
retinaface对人脸的分类需要负样本的补充才能正常训练（必须要）

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [Retinaface (pytorch)](https://github.com/biubug6/Pytorch_Retinaface)
