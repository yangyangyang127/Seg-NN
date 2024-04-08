# No Time to Train: Empowering Non-Parametric Networks for Few-shot 3D Scene Segmentation

This is an old and incomplete version. We will update the code soon ... 

## Introduction
we propose an efficient **N**onparametric **N**etwork for Few-shot 3D **Seg**mentation, Seg-NN, and a further parametric variant, Seg-PN. Seg-NN introduces no learnable parameters and and requires no training. Specifically, Seg-NN extracts dense representations by trigonometric positional encodings and achieves comparable performance to some training-based methods. Building upon Seg-NN, Seg-PN only requires to train a lightweight query-support transferring module (QUEST), which enhances the interaction between the few-shot query and support data.

![framework](framework3d.png)


## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
cd Seg-NN 

conda create -n SegNN python=3.7
conda activate SegNN

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit

pip install -r requirements.txt
pip install pointnet2_ops_lib/.
```

### Datasets

**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**



### Seg-NN 

Seg-NN does not require any training and can conduct few-shot segmentation directly via:

```bash
bash scripts/training_free.sh
```

### Seg-PN 

Train and test our method under few-shot setting:

```bash
bash scripts/training.sh
```

The test procedure has been included in the above command after training and validation.


Note that the above scripts are used for 2-way 1-shot on S3DIS (S_0). Please modify the corresponding hyperparameters to conduct experiments on other settings. 



## Acknowledgement
We thank [Point-NN](https://github.com/ZrrSkywalker/Point-NN/tree/main), [PAP-FZS3D](https://github.com/heshuting555/PAP-FZS3D), and [attMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.
