<div align="center">

# 🔥❄️🌡️ **ICTA2Net**
## **Thinking Aesthetics Assessment of Image Color Temperature**
### _Models, Datasets,   Benchmarks_

[![GitHub Stars](https://img.shields.io/github/stars/chasecjg/ICTA2Net?style=for-the-badge&logo=github&color=ff69b4)](https://github.com/chasecjg/ICTA2Net)
[![GitHub License](https://img.shields.io/github/license/chasecjg/ICTA2Net?style=for-the-badge&color=4169e1)](https://github.com/chasecjg/ICTA2Net/blob/main/LICENSE)
[![AAAI 2026](https://img.shields.io/badge/AAAI%202026-Poster-32cd32?style=for-the-badge)](https://aaai.org/Conferences/AAAI-26/)


</div>

---

## 📢 **Announcement | 项目公告**
> 🎉 **The full paper and official code have been released in this repository.**  
> 🎉 **论文全文与官方代码已正式发布，欢迎查阅与使用！**

---

## 📁 **Resource Links | 项目资源**
| 🔖 Type                | 🔗 Access Link                                                                 |
|------------------------|--------------------------------------------------------------------------------|
| 📄 **Paper / 论文PDF** | [Download PDF](https://github.com/chasecjg/ICTA2Net/blob/main/paper/Thinking%20Aesthetics%20Assessment%20of%20Image%20Color%20Temperature%20Models%2C%20Datasets%20and%20Benchmarks.pdf) |
| 🌐 **Project Page / 项目主页** | [ICTA2Net.github.io](https://chasecjg.github.io/ICTA2Net.github.io/)           |
| 🧠 **Pre-trained Weights / 预训练模型权重** | [Google Drive](https://drive.google.com/file/d/1xpYZbgaj90cSuS5w_yTIQifXjKmkOol_/view?usp=sharing), [百度网盘](https://pan.baidu.com/s/18h99DgDhvC51rt9XW1x5PA?pwd=6666) |
| 📊 **Dataset / 数据集** | [百度网盘](https://pan.baidu.com/s/1lCjeCZ9_CnQlP929PrTyVQ?pwd=6666)                      |
| 📊 **Demo** | [Demo](https://huggingface.co/spaces/AlphaPix/icta2net-demo)                      |

---

## 🖼️ Visualization Gallery  

---

### Dataset Overview  

<p align="center">
  <img src="https://github.com/user-attachments/assets/4145d4f6-7b3e-4965-be27-e52a5b7b2991" 
       alt="Dataset Overview: Color Temperature Distribution & t-SNE Visualization" 
       width="80%">
</p>

<div style="width:88%; margin:auto; text-align:justify; font-size:14px; color:#666;">
<b>Figure 1.</b> Dataset Overview. Our dataset consists of multiple sets of images with different white balance shifts, along with their corresponding high-quality aesthetic reference images. The t-SNE visualization of the images at various color temperatures in the dataset is shown in the figure. This dataset is constructed from linear raw RGB images in the MIT-Adobe FiveK and PPR10K datasets. By precisely simulating the camera ISP process, we generate multiple rendered versions of each image with varying color temperatures.
</div>

<br>

---

### Model Architecture  

<p align="center">
  <img src="https://github.com/user-attachments/assets/62e94d21-db20-41f9-871f-25b507e3b9dc" 
       alt="ICTA2Net Architecture: Cross-Modal Fusion for Color Temperature Aesthetics" 
       width="90%">
</p>

<div style="width:88%; margin:auto; text-align:justify; font-size:14px; color:#666;">
<b>Figure 2.</b> Overall framework of ICTA2Net, comprising four components: a Color Temperature Encoder for capturing color temperature variations; a Contextual Awareness Module (including Visual Encoder, Text Encoder, and Text Denoise Model); a Cross-Modal Fusion Module for visual–textual integration; and a Pairwise Ranking Predictor for aesthetic preference estimation.
</div>

<br>

---

### Aesthetic Ranking Results  

<p align="center">
  <img src="https://github.com/user-attachments/assets/8f9cc0e8-b031-4e38-b88f-efc12b9e84e8" 
       alt="Aesthetic Ranking Visualization: Color Temperature Impact on Image Aesthetics" 
       width="95%">
</p>

<div style="width:88%; margin:auto; text-align:justify; font-size:14px; color:#666;">
<b>Figure 3.</b> Visualization of model ranking results: aesthetic scores decrease progressively from left to right and top to bottom.
</div>



---

## 🚀 Quick Start  
### 1. Environment Preparation
> *Recommended: Python 3.9+, PyTorch 1.12+, CUDA 11.6+*  
```bash
# Clone repository
git clone https://github.com/chasecjg/ICTA2Net.git
cd ICTA2Net
```

### 2. Dataset Setup  

1. 📥 **Download**: Get the dataset from the provided link.  

2. 📂 **Unzip**: Extract to the specified directory (update `dataset_root` in `options.py`).  
3. 📊 **Training Splits**: Two splits are available:  
    - 📜 `train_42.csv`: Full dataset (42k samples)  
    - ⚡ `train_8.csv`: Optimized subset (8k samples, recommended for quick training)  


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
  title={Thinking Aesthetics Assessment of Image Color Temperature: Models, Datasets and Benchmarks},
  author={Cheng, Jinguang and Li, Chunxiao and He, Shuai and Chen, Taiyu and Ming, Anlong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={5},
  pages={3246--3254},
  year={2026}
}
```
---

<div align="left">

### 📫 Welcome to star, fork and collaborate!



<p align="left"><a href="#top">🔝 Back to Top</a></p>

</div>
