# 🔒 Differentially Private Active Learning

Official PyTorch implementation of the SATML 2025 paper "[Differentially Private Active Learning: Balancing Effective Data Selection and Privacy.](https://www.computer.org/csdl/proceedings-article/satml/2025/171100a858/26VnrxNwt9K)" 📄

## 🧠 Abstract

This work introduces **differentially private active learning (DP-AL)** for standard machine learning settings, addressing the previously unexplored challenge of combining active learning with differential privacy. Inspired by **individual privacy accounting**, we propose **step amplification**, a technique that optimizes data utilization by leveraging individual sampling probabilities in batch creation. Through experiments on vision and NLP tasks, we demonstrate that while DP-AL can improve performance on specific datasets, it also reveals fundamental trade-offs between privacy, model accuracy, and data selection effectiveness in privacy-constrained environments.

## 📚 Citing our work

If you find our work useful for your research, please cite our paper as follows:

```bibtex
@INPROCEEDINGS {schwethelm2025dpal,
author = {Schwethelm, Kristian and Kaiser, Johannes and Kuntzer, Jonas and Yigitsoy, Mehmet and Ruckert, Daniel and Kaissis, Georgios},
booktitle = {2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
title = {{ Differentially Private Active Learning: Balancing Effective Data Selection and Privacy }},
year = {2025},
volume = {},
ISSN = {},
pages = {858-878},
doi = {10.1109/SaTML64287.2025.00053},
url = {https://doi.ieeecomputersociety.org/10.1109/SaTML64287.2025.00053},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {apr}
}
```

## 📄 License
This code is released under the [MIT License](LICENSE)

## 🚀 Installation

### 📋 Requirements

- Python>=3.8 🐍

   We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):

   ```bash
   conda create -n DPAL python=3.10 pip 
   conda activate DPAL
   ```

- Install local Opacus package 🔧
    ```bash
    pip install -e opacus
    ```

- PyTorch, torchvision 🔥

   Get the correct command from [here](https://pytorch.org/get-started/locally/). For example, for Linux systems with CUDA version 12.4:
   ```bash
   conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

- Additional requirements 📦
   ```bash
   pip install -r requirements.txt
   ```

### 📊 Dataset preparation

CheXpert and SNLI datasets require manual downloads. For CheXpert, download training and validation images from [kaggle](https://www.kaggle.com/datasets/ashery/chexpert) 📥 and the test images as described in this [repo](https://github.com/rajpurkarlab/cheXpert-test-set-labels) 🐱. Download the SNLI dataset using this [link](https://nlp.stanford.edu/projects/snli/snli_1.0.zip).

For CheXpert, adapt and run [split_resize.py](code/datasets/chexpert/split_resize.py) to resize all images and get the correct data structure. 

### 🤖 Pretrained models

For our CheXpert experiments we use a pretrained NFNet-F0 model. Download the pretrained weights from the [official repository](https://github.com/deepmind/deepmind-research/tree/master/nfnets#pre-trained-weights) and place them inside the folder `code/model/weights` ⚡️.

## 💻 Usage

Run a script file from folder `code/scripts`.