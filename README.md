# 🧠 Brain Tumor CNN-Mamba

A deep learning project for **brain MRI analysis** using three complementary models: a Hybrid CNN-Mamba classifier, a U-Net tumor segmentation model, and a Morphology-Aware CNN-Mamba classifier.

The project progresses from image classification to tumor localization and finally combines learned visual features with explicit tumor morphology features.

---

## 1. 🧠 Project Overview

This project explores the use of **Convolutional Neural Networks (CNNs)** and **Mamba State Space Models** for brain MRI analysis.

Three models are developed:

- **Model 1 — Hybrid CNN-Mamba:** baseline multi-class brain tumor classification.
- **Model 2 — U-Net:** pixel-level tumor segmentation.
- **Model 3 — Morphology-Aware CNN-Mamba:** combines CNN-Mamba visual features with morphology extracted from the Model 2 segmentation mask.

### Overall Pipeline

```text
                         Brain MRI
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌─────────────┐             ┌─────────────┐
       │   MODEL 1   │             │   MODEL 2   │
       │ CNN + Mamba │             │    U-Net    │
       │Classification│            │ Segmentation│
       └─────────────┘             └──────┬──────┘
                                          │
                                          ▼
                                    Tumor Mask
                                          │
                                          ▼
                               Morphology Extraction
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │       MODEL 3       │
                              │ Morphology-Aware    │
                              │     CNN + Mamba     │
                              └──────────┬──────────┘
                                         │
                                         ▼
                                Final Classification
```

---

## 2. 🎯 Objective

The main objectives of the project are:

1. Develop a **CNN-Mamba hybrid architecture** for brain tumor classification.
2. Develop a **U-Net segmentation model** to identify tumor regions in MRI scans.
3. Extract meaningful tumor morphology features from predicted segmentation masks.
4. Combine visual deep-learning features and morphological features.
5. Compare the performance of the baseline CNN-Mamba model with the morphology-aware model.
6. Evaluate the models using appropriate classification and segmentation metrics.

---

## 3. 📊 Dataset Information

The project uses the **BRISC2025** dataset available in the Kaggle environment.

The notebook accesses:

```text
/kaggle/input/brisc2025/brisc2025/
```

The dataset contains separate tasks for:

```text
classification_task/
segmentation_task/
```

### Classification Dataset

The classification training data contains **5,000 MRI images** across four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

### Model 1 Split

Model 1 uses a stratified split of the 5,000 training images:

| Split | Images |
|---|---:|
| Training | 3,500 |
| Validation | 1,000 |
| Testing | 500 |
| **Total** | **5,000** |

### Model 3 Split

Model 3 uses the classification training and test folders:

| Split | Images |
|---|---:|
| Training | 4,000 |
| Validation | 1,000 |
| Test | 1,000 |
| **Total** | **6,000** |

The Model 3 training set is divided into 80% training and 20% validation using a fixed random seed.

### Segmentation Dataset

The segmentation task contains paired:

```text
images/
masks/
```

The notebook keeps only image-mask pairs for which a corresponding mask exists.

The segmentation training data is divided into:

- 80% training
- 20% validation

The provided test split is used for final evaluation.

---

## 4. 🏗️ Three-Model Architecture

### Model 1 — Hybrid CNN-Mamba

```text
Input MRI
   │
   ▼
CNN Feature Extraction
   │
   ├── Conv2D 3 → 32
   ├── BatchNorm
   ├── ReLU
   ├── MaxPool
   │
   ├── Conv2D 32 → 64
   ├── BatchNorm
   ├── ReLU
   └── AdaptiveAvgPool 8×8
   │
   ▼
64 Spatial Tokens × 64 Features
   │
   ▼
Mamba
   │
   ▼
4096 Features
   │
   ▼
Linear Classifier
   │
   ▼
4 Classes
```

Mamba configuration:

```text
d_model = 64
d_state = 16
d_conv  = 4
expand  = 2
```

---

### Model 2 — U-Net

The segmentation network follows the U-Net encoder-decoder architecture.

```text
Input MRI
   │
   ▼
Encoder
32 → 64 → 128 → 256
   │
   ▼
Bottleneck
512 channels
   │
   ▼
Decoder
256 → 128 → 64 → 32
   │
   ▼
1 × 1 Output Convolution
   │
   ▼
Tumor Mask
```

Skip connections connect encoder features to corresponding decoder stages.

The segmentation loss is:

```text
Binary Cross Entropy + (1 - Dice)
```

---

### Model 3 — Morphology-Aware CNN-Mamba

Model 3 combines two feature streams.

#### Visual Stream

```text
MRI
 ↓
CNN
 ↓
64 × 8 × 8 feature map
 ↓
64 spatial tokens
 ↓
Mamba
 ↓
4096 visual features
```

#### Morphology Stream

Six features are extracted from the tumor mask:

```text
1. Tumor Area
2. Tumor Perimeter
3. Tumor Width
4. Tumor Height
5. Aspect Ratio
6. Circularity
```

The morphology features are passed through:

```text
6 → 32 → 32
```

The two streams are then fused:

```text
4096 visual features
        +
32 morphology features
        │
        ▼
4128 features
        │
        ▼
Linear 4128 → 128
        │
       ReLU
        │
   Dropout 0.3
        │
        ▼
Linear 128 → 4
        │
        ▼
Tumor Class
```

### Morphology Equations

Aspect ratio:

```text
Aspect Ratio = Width / Height
```

Circularity:

```text
Circularity = 4π × Area / Perimeter²
```

---

## 5. 📈 Results

### Classification Performance

| Model | Task | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| **Model 1 — Hybrid CNN-Mamba** | Classification | 86.54% | 84.80% | **84.40%** |
| **Model 3 — Morphology-Aware CNN-Mamba** | Classification | 95.75% | 92.00% | **89.50%** |

### Model 1 Test Classification

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 0.80 | 0.73 | 0.76 |
| Meningioma | 0.75 | 0.75 | 0.75 |
| No Tumor | 0.95 | 0.93 | 0.94 |
| Pituitary | 0.88 | 0.95 | 0.91 |
| **Weighted Average** | **0.84** | **0.84** | **0.84** |

### Model 2 Segmentation Performance

| Metric | Validation | Test |
|---|---:|---:|
| Dice | **0.7816** | **0.7915** |
| IoU | **0.6810** | **0.6917** |

Additional test metrics from the notebook:

| Metric | Test Score |
|---|---:|
| Accuracy | **0.9926** |
| Precision | **0.8412** |
| Recall | **0.8027** |
| Specificity | **0.9967** |
| F1 Score | **0.8215** |

> **Note:** Pixel accuracy can be very high when the background occupies most pixels. Dice and IoU are therefore important measures for assessing tumor-region segmentation.

### Model 3 Test Classification

| Metric | Score |
|---|---:|
| Accuracy | **0.8950** |
| Precision | **0.9021** |
| Recall | **0.8950** |
| F1 Score | **0.8949** |

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 0.96 | 0.78 | 0.86 |
| Meningioma | 0.82 | 0.88 | 0.85 |
| Pituitary | 0.97 | 0.96 | 0.97 |
| No Tumor | 0.83 | 0.99 | 0.90 |
| **Weighted Average** | **0.90** | **0.90** | **0.89** |

### Model 1 vs Model 3

```text
Model 1 Test Accuracy = 84.40%
Model 3 Test Accuracy = 89.50%

Improvement = +5.10 percentage points
```

The reported results indicate that adding morphology-aware features to the CNN-Mamba representation improves the classification result over the baseline Model 1 on their respective reported test evaluations.

---

## 6. 🔬 Model 1 / 2 / 3 Descriptions

### Model 1 — Hybrid CNN-Mamba

**Purpose:** Multi-class brain tumor classification.

CNN layers first learn local spatial patterns from the MRI image. The resulting feature map is converted into a sequence of spatial tokens and processed by a Mamba State Space Model. The resulting representation is flattened and passed to a linear classifier.

**Key characteristics:**

- CNN-based local feature extraction
- Mamba-based sequence/spatial dependency modeling
- Four-class classification
- Cross-Entropy loss
- AdamW optimizer
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- 15 training epochs

---

### Model 2 — U-Net

**Purpose:** Tumor-region segmentation.

Model 2 learns to generate a binary tumor mask from an MRI image. The encoder extracts hierarchical features and the decoder reconstructs the tumor region using skip connections.

**Key characteristics:**

- U-Net encoder-decoder
- Four encoder feature levels
- Four decoder stages
- Skip connections
- Binary tumor segmentation
- Binary Cross-Entropy + Dice loss
- AdamW optimizer
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- 20 training epochs

---

### Model 3 — Morphology-Aware CNN-Mamba

**Purpose:** Multi-class tumor classification using both visual and morphological information.

Model 2 generates a tumor mask. Six morphological measurements are calculated from this mask and processed using an MLP. In parallel, the MRI is processed through the CNN-Mamba branch. Both feature representations are concatenated and passed through the final classifier.

**Key characteristics:**

- CNN visual feature extraction
- Mamba sequence modeling
- U-Net-derived tumor mask
- Six morphology features
- Feature-level fusion
- Four-class classification
- Cross-Entropy loss
- AdamW optimizer
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- 15 training epochs
- Dropout: `0.3`

---

## 7. 🛠️ Technologies

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision
- Mamba-SSM

### Image Processing

- OpenCV
- Pillow (PIL)
- NumPy

### Machine Learning / Evaluation

- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Environment

The models were developed and trained in a Kaggle notebook environment with GPU/CUDA support.

### Main Dependencies

```text
torch
torchvision
torchaudio
mamba-ssm
causal-conv1d
numpy
opencv-python
Pillow
scikit-learn
matplotlib
seaborn
```

---

## 8. 📁 Project Structure

The repository is organized as follows:

```text
Brain_tumor_cnn_mamba/
│
├── results/
│   └── experiment results and generated evaluation outputs
│
├── README.md
│
├── ALGORITHMS.md
│
├── RESULTS.md
│
├── brain_tumor_classification.ipynb
│
└── requirements.txt
```

### Model Checkpoints

The trained model files generated by the notebook are:

```text
model1_hybrid_cnn_mamba.pth
model2_unet.pth
model3_morphology_cnn_mamba.pth
```

These checkpoints can be stored separately from the source code when required.

---

## 9. ▶️ Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/SreekarBharadwaz/Brain_tumor_cnn_mamba.git
cd Brain_tumor_cnn_mamba
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-enabled Mamba training, install the compatible PyTorch/CUDA and Mamba dependencies specified by the project environment.

### 4. Dataset

Place the BRISC2025 dataset in the expected directory structure or update the dataset paths in the notebook:

```text
brisc2025/
└── brisc2025/
    ├── classification_task/
    │   ├── train/
    │   └── test/
    │
    └── segmentation_task/
        ├── train/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
```

### 5. Run the notebook

Open:

```text
brain_tumor_classification.ipynb
```

Run the cells in order to:

1. Load and inspect the dataset.
2. Prepare classification data.
3. Train Model 1.
4. Evaluate and save Model 1.
5. Train Model 2.
6. Evaluate and save Model 2.
7. Generate morphology features.
8. Train Model 3.
9. Evaluate and save Model 3.

### Saved Checkpoints

```text
model1_hybrid_cnn_mamba.pth
model2_unet.pth
model3_morphology_cnn_mamba.pth
```

---

## 10. 🚀 Future Improvements

Potential improvements include:

- Add stronger data augmentation.
- Perform systematic hyperparameter optimization.
- Use cross-validation for more robust evaluation.
- Investigate class imbalance and sampling strategies.
- Improve tumor segmentation quality using additional segmentation architectures.
- Compare Mamba against Transformer-based alternatives.
- Perform ablation studies to measure the individual contribution of morphology features.
- Evaluate the model on additional external datasets.
- Add explainability methods such as Grad-CAM or related attribution techniques.
- Build a Streamlit interface for interactive MRI analysis.
- Add automated visualization of predicted tumor masks and morphology measurements.
- Improve checkpoint management and reproducibility.
- Add automated testing and experiment tracking.

---

## 11. ⚠️ Medical-Use Disclaimer

**This project is intended for educational and research purposes only.**

The models presented in this repository are experimental machine-learning systems and **are not medical devices or diagnostic tools**. The predictions, segmentation masks, confidence values, and other outputs should not be used as a substitute for professional medical examination, diagnosis, or treatment.

MRI interpretation should be performed by qualified healthcare professionals using appropriate clinical information and validated diagnostic procedures.

The reported model performance is based on the dataset splits and experimental setup used in this project and does not establish clinical effectiveness or generalization to real-world clinical populations.

---

## 📌 Summary

This project demonstrates a three-stage approach to brain MRI analysis:

```text
Model 1
Hybrid CNN-Mamba
       │
       ▼
Baseline Classification
       │
       │
Model 2
U-Net
       │
       ▼
Tumor Segmentation
       │
       ▼
Morphology Extraction
       │
       ▼
Model 3
Morphology-Aware CNN-Mamba
       │
       ▼
Final Classification
```

### Reported Best Classification Result

**Model 3 — Morphology-Aware CNN-Mamba**

```text
Test Accuracy : 89.50%
Precision      : 90.21%
Recall         : 89.50%
F1 Score       : 89.49%
```

---

## ⭐ Project Highlights

- CNN + Mamba hybrid architecture
- Dedicated U-Net tumor segmentation
- Morphology-aware feature fusion
- Multi-class brain tumor classification
- Quantitative segmentation evaluation
- Classification reports and confusion matrices
- Saved model checkpoints
- Reproducible training configuration

