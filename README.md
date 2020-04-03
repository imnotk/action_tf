# Action-tf

This is a implementation of CCF-Net in TensorFlow. 

## Requirement
* tensorflow 1.4
* dm-sonnet
* Opencv-python

## Dataset preparation 

For [ucf101](https://www.crcv.ucf.edu/data/UCF101.php) and [hmdb51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), we recommend download from origin webset

For Optical flow, we recommend download from [twostreamfusion](https://github.com/feichtenhofer/twostreamfusion),
or you can generate your own optical flow dataset by following the [TSN](https://github.com/yjxiong/temporal-segment-networks) setting

## Training and Testing

To train a new model, all my training setting is under the root of "script".

```bash
bash ts_rgb_resnet50.sh
bash ts_flow_resnet50.sh
bas_resnet50_non_local_fusion.sh
```

## Eval Score

Use the following command to test its performance of ucf101:

```bash
python eval_score_resnet50.py -modality fusion --dataset UCF101
```

## Minor difference
原始TSN使用pytorch和caffer实现，由于用tensorflow实现可能会存在一定的性能差异,以下时split1上的性能
| Modality | ResNet50 | ResNet101 |
| :-----| ----: | :----: |
| RGB  | 84.6%~84.8% | 86.3%~87.2% |
| Flow | 87.2%~87.4% | 88.2%~88.4% |
| CCF-Net | 93.6%~93.8% | 94.4%~94.6% |

## confusion matrix
每次使用train.py和test.py后会在相应的文件夹中生成准确率等on-the-fly文件如：
```
logdir/UCF101/0/TS_resnet50/rgb/video_predict_multi.pickle
logdir/UCF101/0/TS_resnet50/rgb/train_log.txt
logdir/UCF101/0/TS_resnet50/rgb/val_log.txt
logdir/UCF101/0/TS_resnet50/rgb/test_result_multi.txt
logdir/UCF101/0/TS_resnet50/rgb/test_result_center.txt
```

随后生成混淆矩阵
```
python ./script/confusion_matrix_ts_resnet50.py
```

## 重要通知
由于tensorflow1.x没有自带的ImageNet预训练权重，我们需要从Tensorflow Models里面下载slim的模型，再转换成dm-sonnet的模型，需要你先从[slim](https://github.com/tensorflow/models/tree/master/research/slim)中下载ResNet50等模型权重(不要下载成TF2的)，然后使用utils目录下的rebuild_ckpt.py文件转换模型。
注意：我没有提供argparser等命令行指令，需要你手动修改你的模型权重路径。
```
python rebuild_ckpt.py
```
对于数据集同理，由于没有提供整体的config文件，你需要在Dataset里面调整各个文件的路径，这只是个Draft文件。