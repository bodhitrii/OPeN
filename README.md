## DAR-BN

Shiran Zada, Michal Irani

------

This is the unofficial implementation of DAR-BN in the paper [Pure Noise to the Rescue of Insufficient Data: Improving Imbalanced Classification by Training on Random Noise Images](https://arxiv.org/pdf/2112.08810.pdf) (NeurIPS 2022 spotlight) in pytorch.

No official implementation code yet found.


### How to run the code

**1. Setting up the python environment**

- PyTorch 1.2
- TensorboardX
- scikit-learn

\
**2. Dataset**

- Imbalanced CIFAR. The original data will be downloaded and converted by imbalancec_cifar.py.

\
**3. Run the code on CIFAR-10/100 long-tail datasets**

Use the following command to run the code on CIFAR-10/100 long-tail datasets.
For comparison, other loss_type & train_rule(train methods) are prepared in the code. 

- To train the ERM baseline on long-tailed imbalance with ratio of 100

```javascript
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --arch wide_resnet28_10
```

- To train the ERM Loss along with DAR-BN on long-tailed imbalance with ratio of 100
```javascript
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule DAR-BN --arch wide_resnet28_10
```


### Results
