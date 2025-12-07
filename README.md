<div align="center">

# 🌡️ ICTA2Net: Thinking Aesthetics Assessment of Image Color Temperature  
## Models, Datasets, and Benchmarks  

[![GitHub Stars](https://img.shields.io/github/stars/chasecjg/ICTA2Net?style=for-the-badge&color=ff69b4)](https://github.com/chasecjg/ICTA2Net)
[![GitHub License](https://img.shields.io/github/license/chasecjg/ICTA2Net?style=for-the-badge&color=4169e1)](https://github.com/chasecjg/ICTA2Net/blob/main/LICENSE)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-Poster-blue?style=for-the-badge&color=32cd32)](https://aaai.org/Conferences/AAAI-26/)

*Elegance in color temperature, precision in aesthetic assessment*  

</div>

---

## 📢 Announcement  
> Thank you for your attention! The full paper and official code have been released in this repository.  
> 感谢各位同仁的关注，论文全文与官方代码已在本仓库发布。  

### 📁 Resource Links  
| Resource Type | Access Link |  
|---------------|-------------|  
| 📜 Paper | [PDF](https://github.com/chasecjg/ICTA2Net/blob/main/paper/Thinking%20Aesthetics%20Assessment%20of%20Image%20Color%20Temperature%20Models%2C%20Datasets%20and%20Benchmarks.pdf) |  
| 🌐 Project Page | [ICTA2Net.github.io](https://chasecjg.github.io/ICTA2Net.github.io/) |  
| ⚖️ Pre-trained Weights | [Google Drive](https://drive.google.com/file/d/1xpYZbgaj90cSuS5w_yTIQifXjKmkOol_/view?usp=sharing) / [百度网盘](https://pan.baidu.com/s/18h99DgDhvC51rt9XW1x5PA?pwd=6666) |  
| 📊 Dataset | [百度网盘](https://pan.baidu.com/s/1xxxxxx) *(待补充)* |  

---

## 🖼️ Visualization Gallery  

### Dataset Overview  
<div align="center">
  <img src="https://github.com/user-attachments/assets/4145d4f6-7b3e-4965-be27-e52a5b7b2991" 
       alt="Dataset Overview: Color Temperature Distribution & t-SNE Visualization" 
       width="80%" 
       style="border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <p style="font-size: 14px; color: #666; margin-top: 8px;">Figure 1: Dataset Overview. Our dataset consists of multiple sets of images with different white balance shifts, along with their corresponding high-quality aesthetic reference images. The t-SNE visualization of the images at various color temperatures in the dataset is shown in the figure. This dataset is constructed from linear raw RGB images in the MIT-Adobe FiveK and PPR10K datasets. By precisely simulating the camera ISP process, we generate multiple rendered versions of each image with varying color temperatures.</p>
</div>

<br>

### Model Architecture  
<div align="center">
  <img src="https://github.com/user-attachments/assets/62e94d21-db20-41f9-871f-25b507e3b9dc" 
       alt="ICTA2Net Architecture: Cross-Modal Fusion for Color Temperature Aesthetics" 
       width="90%" 
       style="border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <p style="font-size: 14px; color: #666; margin-top: 8px;">Figure 2: Overall framework of ICTA2Net, comprising four components: a Color Temperature Encoder for capturing color temperature variations; a Contextual Awareness Module (including Visual Encoder, Text Encoder, and Text Denoise Model); a Cross-Modal Fusion Module for visual-textual integration; and a Pairwise Ranking Predictor for aesthetic preference estimation.</p>
</div>

<br>

### Aesthetic Ranking Results  
<div align="center">
  <img src="https://github.com/user-attachments/assets/8f9cc0e8-b031-4e38-b88f-efc12b9e84e8" 
       alt="Aesthetic Ranking Visualization: Color Temperature Impact on Image Aesthetics" 
       width="95%" 
       style="border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
  <p style="font-size: 14px; color: #666; margin-top: 8px;">Figure 3: Visualization of model ranking results: aesthetic scores decrease progressively from left to right and top to bottom.</p>
</div>

---

## 🚀 Quick Start  

### 1. Environment Preparation  
> *Recommended: Python 3.9+, PyTorch 1.12+, CUDA 11.6+*  

```bash
# Clone repository
git clone https://github.com/chasecjg/ICTA2Net.git
cd ICTA2Net

# Install dependencies
pip install -r requirements.txt
```


### 2. Dataset Setup
1. Download the dataset from the provided link.
2. Unzip to the specified directory (modify dataset_root in options.py).
3. Two training splits are provided:
  - train_42.csv: Full dataset (42k samples)
  - train_8.csv: Optimized subset (8k samples, recommended for quick training)

### 3. Model Training
```bash
# Modify hyperparameters in options.py (e.g., resume, weight path)
python train.py
```
### 4. Inference & Evaluation
```bash
# Adjust test parameters in options.py (e.g., test dataset path)
python test.py
```

### 📝 Citation
```bibtex
@inproceedings{cheng2026thinking,
  title     = {Thinking Aesthetics Assessment of Image Color Temperature: Models, Datasets and Benchmarks},
  author    = {Cheng, Jinguang and Li, Chunxiao and He, Shuai and Chen, Taiyu and Ming, Anlong},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026},
  note      = {Poster},
  url       = {https://github.com/chasecjg/ICTA2Net/tree/main}
}
```