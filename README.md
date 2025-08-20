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
--data=/root/stylegan3-main/dataset.zip \
--gpus=1 \
--batch=8 \
--kimg=100 \
--gamma=8.2 \
--batch-gpu=8 \
--snap=2 \
--mirror=1 \
--resume=/root/stylegan3-main/weight_files/stylegan3-r-afhqv2-512x512.pkl
```
（--outdir: Training output directory
--cfg: Model Configuration (stylegan3-r)
--data: Training dataset path
--gpus: The number of Gpus used
--batch: Batch size
--kimg: Training duration (thousand images)
--gamma: R1 regularized weight
--batch-gpu: The batch size of each GPU
--snap: Snapshot saving frequency
--mirror: Data Augmentation (1 enabled)
--resume: Pre-trained weight path）

Generation and Visualization
After the training is completed, use the visualizer.py tool to generate and visualize images:
```bash
python visualizer.py --pkl /path/to/your/DL-BBI-000028.pkl
```

Calculate the FID value
After the training is completed, the FID score is calculated using the final generated.PKL file to quantitatively evaluate the quality of the generated images.
```bash
cd /path/to/stylegan3-main
python calc_metrics.py --metrics=fid50k_full \
                       --data="/path/to/your/dataset.zip" \
                       --network="./training-runs/network-snapshot-000028.pkl" \
                       --gpus=1
```

Results and Demonstration
This project successfully generated a smooth transition sequence from the initial form of the product to the bionic biological form. Each frame of the image maintains high fidelity and coherent morphological changes, demonstrating the powerful capabilities of StyleGAN3 in cross-domain image generation and morphological innovation.

[Fig13.TIF.TIF](https://github.com/user-attachments/files/21891843/Fig13.TIF.TIF)

License
Based on the NVIDIA StyleGAN3 license, please read the official license terms before use.

Acknowledgments
StyleGAN3 models provided by NVIDIA Research
GPU computing power support provided by the AutoDL platform
