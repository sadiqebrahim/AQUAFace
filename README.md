<div align="center">

# *AQUAFace* : Age-Invariant Quality Adaptive Face Recognition for Unconstrained Selfie vs ID Verification  
<h3><strong>AAAI 2025</strong></h3>

Shivang Agarwal<sup>*1</sup>  &emsp; Jyoti Chaudhary<sup>*1</sup>  &emsp; Sadiq Siraj Ebrahim<sup>1$</sup>  &emsp; Mayank Vatsa<sup>1</sup>  &emsp; 

Richa Singh<sup>1</sup>  &emsp; Shyam Prasad Adhikari<sup>2+</sup>  &emsp; Sangeeth Reddy Battu<sup>2+</sup> &emsp;   

<sup>1</sup>IIT Jodhpur, <sup>2</sup>Swiggy

<sup>*</sup>Equal Contribution, <sup>+</sup>Work done while at Swiggy, 
<sup>$</sup>Work done as a part of summer internship at IITJ

<a href='https://sadiqebrahim.github.io/AQUAFace/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://ojs.aaai.org/index.php/AAAI/article/view/32165'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<!---<a href='https://huggingface.co/your-model-link'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>
--->

</div>
<hr />

## Contributions

**AQUAFace** is a novel face recognition framework that adapts to **age variations** and **image quality discrepancies** — two major challenges in selfie vs ID verification. It introduces the **Age and Quality Likelihood Ratio (AQUALR)** to enhance model performance across diverse domains.

The key contributions of our work are,<br>
1️⃣ AQUALR: A Gaussian Mixture Model (GMM)-based module computes an Age and Quality Likelihood Ratio (AQUALR), integrating age and quality labels into pairwise similarity scores. This enables dynamic sample weighting based on age and quality variations.<br>
2️⃣ Adaptive Contrastive Loss: The loss dynamically adjusts margins using AQUALR to penalize harder samples characterized by large age differences or low quality, enhancing intra-class compactness and inter-class separation.<br>
3️⃣ Identity Preservation: A fine-tuned ArcFace model ensures robust identity-related feature extraction, utilizing margin-based softmax loss to maintain discriminative power under diverse variations. The combined loss function integrates these components to optimize for both identity preservation and resilience against age and quality changes. The framework incorporates real and synthetic datasets for training, using synthetic data fine-tuning to enrich intra-class variability. The architecture leverages a Siamese network with shared weights and cosine similarity for feature comparison, achieving state-of-the-art performance across benchmark datasets.


> **<p align="justify"> Abstract:** *Face recognition in the presence of age and quality variations poses a formidable challenge. While recent margin-based loss functions have shown promise in addressing these variations individually, real-world scenarios such as selfie versus ID face matching often involve simultaneous variations of both age and quality. In response, we propose a comprehensive framework aimed at mitigating the impact of these variations while preserving vital identity-related information crucial for accurate face recognition. The proposed adaptive margin-based loss function AQUAFace exhibits adaptiveness towards hard samples characterized by significant age and quality variations. This loss function is meticulously designed to prioritize the preservation of identity-related features while simultaneously mitigating the adverse effects of age and quality variations on face recognition accuracy. To validate the effectiveness of our approach, we focus on the specific task of selfie versus ID document matching. Our results demonstrate that AQUAFace effectively handles age and quality differences, leading to enhanced recognition performance. Additionally, we explore the benefits of fine-tuning the recognition model with synthetic data, further boosting performance. As a result, our proposed model, AQUAFace, achieves state-of-the-art performance on six benchmark datasets (CALFW, CPLFW, CFP-FP, AgeDB, IJB-C, and TinyFace), each exhibiting diverse age and quality variations.* </p>

## Framework

<p align="center" width="100%">
  <img src='docs/static/images/model.jpeg' height="75%" width="75%">
</p>

Figure 2. Training pipeline of AQUAFace. We have introduced a novel adaptive margin-based loss for age-invariant, quality-aware face recognition, specifically targeting selfie vs. ID verification tasks.

---


## Installation
```bash
conda create -n aquaface python=3.10
conda activate aquaface
pip install -r requirements.txt
```

---

## Datasets

### Face Recognition Benchmarks

To evaluate AQUAFace on standard datasets:

| Dataset | Download Link | Notes |
|--------|----------------|-------|
| AgeDB | [Link](http://datasets.com/agedb) | Aligned face images |
| CFP-FP | [Link](http://datasets.com/cfp) | Frontal vs profile |
| CALFW | [Link](http://datasets.com/calfw) | Age variation |
| CPLFW | [Link](http://datasets.com/cplfw) | Pose variation |
| IJB-C | [Link](https://nvlabs-fi-cdn.nvidia.com/ijbc/) | Real-world benchmark |
| TinyFace | [Link](https://github.com/Tencent/TinyFace) | Low-resolution faces |

Please align and crop faces using [MTCNN](https://github.com/ipazc/mtcnn) or [InsightFace](https://github.com/deepinsight/insightface) before use.

### Synthetic Data (SynAM)

- [SynAM Download (20GB)](https://drive.google.com/drive/folders/XXXXX)


---


## Data Preparation
Please arrange your dataset in the following structure:
```bash
dataset/
├── train/
│   ├── id/
│   ├── selfie/
├── val/
│   ├── id/
│   ├── selfie/
```

---

## Training
```bash
python train.py \
  --config configs/aquaface_config.yaml \
  --output_dir ./logs/aquaface_run \
  --data_path ./dataset
```

---

## Evaluation
```bash
python evaluate.py \
  --checkpoint ./logs/aquaface_run/best_model.pth \
  --data_path ./dataset/val
```

---

## Results

| Model      | Dataset       | Protocol   | Accuracy (%) |
|------------|---------------|------------|--------------|
| Baseline   | [YourDataset] | Cross-Age  | 85.2         |
| AQUAFace   | [YourDataset] | Cross-Age  | **91.7**     |
| AQUAFace   | [YourDataset] | Cross-Domain | **92.3**    |

---



## 📈 Results

| Dataset  | Accuracy (%) |
|----------|--------------|
| CALFW    | 93.2         |
| CPLFW    | 91.8         |
| CFP-FP   | 96.5         |
| AgeDB    | 94.7         |
| IJB-C    | 95.1         |
| TinyFace | 87.4         |

---

## 🖼️ Visualizations

<table>
<tr>
  <td><img src="assets/vis_age.png" width="300"></td>
  <td><img src="assets/vis_quality.png" width="300"></td>
</tr>
<tr>
  <td align="center">Age Invariance</td>
  <td align="center">Quality Adaptiveness</td>
</tr>
</table>

## Citation
```bibtex
@inproceedings{agarwal2025AQUAFace,
  title={{AQUAFace}: Age-Invariant Quality Adaptive Face Recognition for Unconstrained Selfie vs ID Verification},
  author={Shivang Agarwal and Jyoti Chaudhary and Sadiq Siraj Ebrahim and Mayank Vatsa and Richa Singh and Shyam Prasad Adhikari and Sangeeth Reddy Battu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2025}
}
```

