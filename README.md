# Differentially Private Active Learning

[![Paper](https://img.shields.io/badge/Paper-SaTML%202025-blue)](https://www.computer.org/csdl/proceedings-article/satml/2025/171100a858/26VnrxNwt9K)
[![arXiv](https://img.shields.io/badge/arXiv-2410.00542-b31b1b.svg)](https://arxiv.org/abs/2410.00542)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the SATML 2025 paper "[Differentially Private Active Learning: Balancing Effective Data Selection and Privacy.](https://www.computer.org/csdl/proceedings-article/satml/2025/171100a858/26VnrxNwt9K)"

## Abstract

This work introduces **differentially private active learning (DP-AL)** for standard machine learning settings, addressing the previously unexplored challenge of combining active learning with differential privacy. Inspired by **individual privacy accounting**, we propose **step amplification**, a technique that optimizes data utilization by leveraging individual sampling probabilities in batch creation. Through experiments on vision and NLP tasks, we demonstrate that while DP-AL can improve performance on specific datasets, it also reveals fundamental trade-offs between privacy, model accuracy, and data selection effectiveness in privacy-constrained environments.

## Citing our work

If you find our work useful for your research, please cite our paper as follows:

```bibtex
@INPROCEEDINGS {schwethelm2025dpal,
author = {Schwethelm, Kristian and Kaiser, Johannes and Kuntzer, Jonas and Yigitsoy, Mehmet and Ruckert, Daniel and Kaissis, Georgios},
booktitle = {2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
title = {{Differentially Private Active Learning: Balancing Effective Data Selection and Privacy}},
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

## License
This code is released under the [MIT License](LICENSE)

## Installation

### Setting Up the Environment

#### Using UV Package Manager

[UV](https://docs.astral.sh/uv/) is a fast dependency resolver and Python package manager that we use for this project.

1. **Install UV**
   - Follow the installation instructions at the [UV documentation](https://docs.astral.sh/uv/)
   - Quick install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. **Setup Project Environment**
   - Clone this repository: `git clone https://github.com/kschwethelm/DPAL`
   - Navigate to the project directory: `cd DPAL`

3. **Manage Dependencies**
   - Install dependencies:
     ```bash
     uv sync
     ```

4. **Activate the Environment**
   ```bash
   source .venv/bin/activate
   ```

### Dataset preparation

The CheXpert dataset requires manual download. Download training and validation images from [kaggle](https://www.kaggle.com/datasets/ashery/chexpert) and the test images as described in this [repo](https://github.com/rajpurkarlab/cheXpert-test-set-labels).

To resize all images and get the correct data structure, adapt dataset path in [code/datasets/chexpert/split_resize.py](code/datasets/chexpert/split_resize.py) and run the script. For memmap creation, adapt dataset path in [code/datasets/chexpert/create_memmap.py](code/datasets/chexpert/create_memmap.py) and run the script.

### Pretrained models

For our CheXpert experiments we use a pretrained NFNet-F0 model. Download the pretrained weights from the [official repository](https://github.com/deepmind/deepmind-research/tree/master/nfnets#pre-trained-weights) and place them inside the folder `code/models/weights`.

## Usage

Run a script file from folder `code/scripts`.