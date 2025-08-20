# DL-BIB
DL-BIB Method: Intelligent Generation of Product Bionic Image Forms via Multimodal Emotion-Weighted Quantification
This project is based on the StyleGAN3 model officially released by NVIDIA and has achieved progressive image fusion and generation. Train on the AutoDL cloud platform using the RTX 4090D GPU to generate images of a specific style by fine-tuning the pre-trained model.

Environmental configuration：
1、Operating system: Ubuntu 18.04 LTS
2、GPU: NVIDIA RTX 4090D (24GB VRAM)
3、Python: 3.12
4、PyTorch: 2.3.0
5、CUDA: 12.1
6、Environmental Management: Conda

Installation steps：
1、Clone the official code repository：
```bash
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
```
2、Create a conda environment and install dependencies (refer to the official README)
3、Prepare the dataset and package the images into a ZIP file:
```bash
zip -r dataset.zip ./images/
```

Training command
1、Use the following command to train (fine-tune) the model:
```bash
python train.py --outdir=./training-runs \
--cfg=stylegan3-r \
--data=/root/stylegan3-main/dog512x512.zip \
--gpus=1 \
--batch=8 \
--kimg=100 \
--gamma=8.2 \
--batch-gpu=8 \
--snap=2 \
--mirror=1 \
--resume=/root/stylegan3-main/weight_files/stylegan3-r-afhqv2-512x512.pkl
```
