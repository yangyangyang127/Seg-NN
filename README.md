# No Time to Train 🚀: Empowering Non-Parametric Networks for Few-shot 3D Scene Segmentation


## 💥 News

- **[2024.04]** Seg-NN is awarded as 🔥 ***Highlight Paper*** 🔥 in CVPR 2024!
- **[2024.02]** Seg-NN is accepted by CVPR 2024 🎉!
- **[2023.12]** We release the [paper](https://arxiv.org/pdf/2404.04050.pdf) adapting [Point-NN & Point-PN](https://github.com/ZrrSkywalker/Point-NN) into 3D scene segmentation tasks.

## Introduction
we propose an efficient **N**onparametric **N**etwork for Few-shot 3D **Seg**mentation, Seg-NN, and a further parametric variant, Seg-PN. Seg-NN introduces no learnable parameters and requires no training. Specifically, Seg-NN extracts dense representations by trigonometric positional encodings and achieves comparable performance to some training-based methods. Building upon Seg-NN, Seg-PN only requires to train a lightweight query-support transferring module (QUEST), which enhances the interaction between the few-shot query and support data.

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

**Please Note:** We find a [bug](https://github.com/yangyangyang127/Seg-NN/blob/main/dataloaders/loader.py#L54) in existing data preparation codes. The pre-processed points via existing code are ordered, which may cause models to learn the order of points during training. Thus we add [this line](https://github.com/yangyangyang127/Seg-NN/blob/eddbbedc532abe3f02e362f846c98f24f7fb4ea9/dataloaders/loader.py#L58) to shuffle the points.



### Seg-NN 

Seg-NN does not require any training and can conduct few-shot segmentation directly via:

```bash
bash scripts/segnn.sh
```

### Seg-PN 

We have released the pre-trained models under [log_s3dis_SegPN](https://github.com/yangyangyang127/Seg-NN/tree/main/log_s3dis_SegPN) and [log_scannet_SegPN](https://github.com/yangyangyang127/Seg-NN/tree/main/log_scannet_SegPN) fold. To test our model, direct run:

```bash
bash scripts/segpn_eval.sh
```

Please note that randomness exists during training even though we have set a random seed.

If you want to train our method under the few-shot setting:

```bash
bash scripts/segpn.sh
```

The test procedure has been included in the above training command after validation.


Note that the above scripts are used for 2-way 1-shot on S3DIS (S_0). Please modify the corresponding hyperparameters to conduct experiments in other settings. 



## Acknowledgement
We thank [Point-NN](https://github.com/ZrrSkywalker/Point-NN/tree/main), [PAP-FZS3D](https://github.com/heshuting555/PAP-FZS3D), and [attMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.
