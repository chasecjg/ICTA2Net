# Thinking Aesthetics Assessment of Image Color Temperature: Models, Datasets, and Benchmarks

- Thank you for your attention. The paper has been uploaded to the repository, and the code will be released around the end of November. 
- 感谢各位同仁们的关注，论文已上传至该仓库，代码将在11月底左右发布。
- [paper](https://github.com/chasecjg/ICTA2Net/blob/main/paper/Thinking%20Aesthetics%20Assessment%20of%20Image%20Color%20Temperature%20Models%2C%20Datasets%20and%20Benchmarks.pdf)
- [project](https://chasecjg.github.io/ICTA2Net.github.io/)
 
### Dataset Overview
<div align="center">
  <img src="https://github.com/user-attachments/assets/4145d4f6-7b3e-4965-be27-e52a5b7b2991" 
       alt="Dataset Overview" 
       width="600">
</div>

Figure 1: Dataset Overview. Our dataset consists of multiple sets of images with different white balance shifts, along with their corresponding high-quality aesthetic reference images. The t-SNE visualization of the images at various color temperatures in the dataset is shown in the figure. This dataset is constructed from linear raw RGB images in the MIT-Adobe FiveK and PPR10K datasets. By precisely simulating the camera ISP process, we generate multiple rendered versions of each image with varying color temperatures.

---

### Model Architecture Diagram
<div align="center">
  <img src="https://github.com/user-attachments/assets/62e94d21-db20-41f9-871f-25b507e3b9dc" 
       alt="Model Architecture Diagram" 
       width="1151">
</div>

Figure 2: Overall framework of ICTA2Net, comprising four components: a Color Temperature Encoder for capturing color temperature variations; a Contextual Awareness Module (including Visual Encoder, Text Encoder, and Text Denoise Model); a Cross-Modal Fusion Module for visual-textual integration; and a Pairwise Ranking Predictor for aesthetic preference estimation.

---

### Visualization Results
<div align="center">
  <img src="https://github.com/user-attachments/assets/8f9cc0e8-b031-4e38-b88f-efc12b9e84e8" 
       alt="Visualization Results" 
       width="1200">
</div>

Figure 3: Visualization of model ranking results: aesthetic scores decrease progressively from left to right and top to bottom.


## If you find our work is useful, please cite our paper:
```
@inproceedings{cheng2026thinking,
  title     = {Thinking Aesthetics Assessment of Image Color Temperature: Models, Datasets and Benchmarks},
  author    = {Cheng, Jinguang and Li, Chunxiao and He, Shuai and Chen, Taiyu and Ming, Anlong},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026},
  note      = {Poster},
  url       = {https://github.com/chasecjg/ICTA2Net/tree/main}
}
```
