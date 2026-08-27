# Model 3 — Morphology-Aware CNN-Mamba

## Overview

Model 3 is the advanced brain tumor classification model developed in this project.

It combines:

- U-Net-based tumor segmentation
- CNN-based visual feature extraction
- Mamba-based spatial feature modeling
- Tumor morphological feature extraction
- Feature fusion
- Multi-class classification

The main idea is to combine **learned visual features** with **explicit morphological information** extracted from the segmented tumor region.

---

## Overall Architecture

```text
                         Brain MRI Image
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
             U-Net                            CNN
          Segmentation                  Feature Extraction
                │                               │
                ▼                               ▼
          Tumor Mask                         Mamba
                │                               │
                ▼                               ▼
     Morphological Features             4096 Features
                │                               │
                ▼                               │
        Morphology MLP                         │
                │                               │
                ▼                               │
           32 Features                          │
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                         Feature Fusion
                                │
                                ▼
                           4128 Features
                                │
                                ▼
                        Classification Head
                                │
                                ▼
                         4 Tumor Classes
