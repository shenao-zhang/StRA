# Structure-Regularized Attention 
Code for paper "Structure-Regularized Attention for Deformable Object Representation". 

The code contains the ResNet50-StRA network for person re-identification task and the corresponding configurations to reproduce our results. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Training

The reported results are trained and evaluated based on the existing person ReID framework: https://github.com/KaiyangZhou/deep-person-reid.

To reproduce the reported results, please pull from the above `deep-person-reid` repository (version 0.9.1)
and download the datasets following the instructions. Then integrate with our network code:

```
mv models ./deep-person-reid/torchreid
```
To train our StRA-ResNet50 on Market-1501 dataset:
 
```
python train.py
```
 To train on different datasets, change different dataset sources in `train.py`.
 
## Evaluation

Simply running the following code will give the reported results on Market-1501 dataset:
```
python eval.py
```

## Results

By downloading our pre-trained models and running the evaluation code, the followng results will be obtained:

| Dataset      | mAP  | rank1 |rank5 |rank10 |
| ------------------ |---------------- | -------------- |-------------- |------------- |
| Market1501  |     84.2%         |      94.0%       |  97.6% |98.5% |

## Pre-trained Models
 The checkpoint model on Market-1501 dataset can be found at https://drive.google.com/file/d/1oXLY60iX8Vkbp-iTlrrzzkrFjME1Esun/view?usp=sharing.
